/*
 * usbfs_trace.c - an LD_PRELOAD ioctl interposer that records, but never
 * alters, the USBDEVFS traffic one process issues to exactly one usbfs node.
 *
 * Why an interposer at all: the vendor SDK reaches this scanner entirely
 * through ioctl() on /dev/bus/usb/BBB/DDD (its statically linked libusb never
 * touches /dev/video* or /dev/hidraw*), and it uses the asynchronous URB model
 * (USBDEVFS_SUBMITURB + USBDEVFS_REAPURB*). A plain strace proves *that*
 * transfers happened and on which fd, but prints the struct usbdevfs_urb
 * argument as a bare pointer, so the wire bytes are not recoverable from it.
 * Running inside the process, we can dereference that pointer.
 *
 * Design rules this file is built to, in priority order:
 *
 *   1. Observe, never alter. Every hooked call is forwarded to the real libc
 *      symbol with its arguments untouched, and the caller's return value and
 *      errno are restored verbatim. The tracer issues no device operation of
 *      its own - no ioctl, no read, no write against the traced fd.
 *   2. Opt in, and fail closed. Capture happens only when both
 *      SCANNER3D_TRACE_TARGET and SCANNER3D_TRACE_OUT are set. If the target
 *      is absent, is not a character device under /dev/bus/usb, or the output
 *      file already exists, the tracer disables itself and records nothing.
 *      "Disabled" always means "captures nothing", never "captures everything".
 *   3. One node snapshot only. An fd is traced only if it was opened on the
 *      exact canonical target path *and* its st_rdev matches the target's.
 *      File descriptors for other USB devices, other files, and the tracer's
 *      own output are never recorded. Path plus rdev scopes the capture to the
 *      node resolved at startup; it cannot prove physical identity across a
 *      hotplug - see the comment above consider_fd().
 *   4. Everything is bounded. Per-event payload bytes, per-line length, and
 *      total output bytes all have hard caps, and hitting one is written into
 *      the stream as evidence rather than silently dropping data.
 *   5. Reading user memory cannot crash the traced process. Payloads are read
 *      with process_vm_readv() against our own pid, so a stale or unmapped
 *      buffer pointer yields EFAULT instead of SIGSEGV.
 *
 * Output is line-delimited JSON: one header object, then one object per
 * recorded call. See reference/tracer/README.md for the schema and
 * src/scanner3d/usbfs_trace.py for the offline decoder that consumes it.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/ioctl.h>
#include <linux/usbdevice_fs.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define TRACE_FORMAT "scanner3d-usbfs-trace"
#define TRACE_VERSION 1

/* Environment variables. All capture is opt-in through these. */
#define ENV_TARGET "SCANNER3D_TRACE_TARGET"
#define ENV_OUT "SCANNER3D_TRACE_OUT"
#define ENV_MAX_PAYLOAD "SCANNER3D_TRACE_MAX_PAYLOAD"
#define ENV_MAX_BYTES "SCANNER3D_TRACE_MAX_BYTES"

/* Hard caps. The env vars may lower these but never raise them. */
#define PAYLOAD_CAP 4096u
#define PAYLOAD_DEFAULT 256u
#define OUTPUT_CAP (256u * 1024u * 1024u)
#define OUTPUT_DEFAULT (32u * 1024u * 1024u)

/*
 * A capture is only evidence if it can be read back, and the decoder requires
 * a header first. So the output cap has a floor as well as a ceiling: enough
 * room for the header record and, kept in reserve beneath the cap, the two
 * records that explain a capture which stopped early - the ``limit`` record
 * that says the cap was reached, and the closing ``summary`` that says how
 * many events were lost to it. Ordinary events may not spend the reserve, so
 * those two always fit. A configured cap below the floor is raised to it and
 * the request is written into the header, so a clamp is visible in the
 * evidence rather than silently applied. The result is that every file this
 * tracer creates is a parseable stream that accounts for its own losses - in
 * the smallest case a header, a limit and a summary - never a truncated
 * fragment.
 */
#define OUTPUT_MIN (8u * 1024u)
#define LIMIT_RESERVE 1024u

/* The reserve is only a guarantee if it is larger than the two records it
 * reserves room for, and if the smallest permitted cap leaves room for events
 * as well as the reserve. Both are checked at compile time, not trusted. */
_Static_assert(LIMIT_RESERVE >= 768u, "reserve must hold the limit and summary records");
_Static_assert(OUTPUT_MIN > 2u * LIMIT_RESERVE, "smallest cap must leave room for events");

/* One JSON line is assembled in this fixed buffer; 2 hex chars per payload
 * byte for up to two payload fields, plus generous room for scalar fields. */
#define LINE_CAP (4u * PAYLOAD_CAP + 2048u)

/* fds at or above this are not tracked; the omission is announced in-stream. */
#define FD_TABLE_SIZE 4096

/* ------------------------------------------------------------------ state */

static int (*real_ioctl)(int, unsigned long, ...);
static int (*real_open)(const char *, int, ...);
static int (*real_open64)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_openat64)(int, const char *, int, ...);
static int (*real___open_2)(const char *, int);
static int (*real___open64_2)(const char *, int);
static int (*real___openat_2)(int, const char *, int);
static int (*real___openat64_2)(int, const char *, int);
static int (*real_close)(int);
static int (*real_dup)(int);
static int (*real_dup2)(int, int);
static int (*real_dup3)(int, int, int);

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;

static int enabled;             /* capture is armed */
static int out_fd = -1;         /* output file, opened O_EXCL */
static char target_path[PATH_MAX];
static char out_prefix[PATH_MAX];   /* SCANNER3D_TRACE_OUT, before the .pid suffix */
static char out_path[PATH_MAX];     /* what this process actually writes */
static dev_t target_rdev;
static unsigned max_payload = PAYLOAD_DEFAULT;
static unsigned long max_bytes = OUTPUT_DEFAULT;
/* What the environment asked for, before clamping; equal to the above when no
 * clamp happened. Reported in the header so the two can be compared. */
static unsigned long long requested_payload = PAYLOAD_DEFAULT;
static unsigned long long requested_bytes = OUTPUT_DEFAULT;

static unsigned long written_bytes;   /* out_lock */
static unsigned long long seq_counter; /* out_lock */
static unsigned long dropped_events;   /* out_lock: over cap or line overflow */
static int limit_announced;            /* out_lock */
static int output_failed;              /* out_lock: creation was tried and failed */
static int fd_overflow_announced;      /* out_lock */

/* Written under out_lock; read without it on the ioctl fast path, so the
 * element type has to make that read well defined rather than merely
 * benign-looking. */
static _Atomic unsigned char tracked_fd[FD_TABLE_SIZE];

/* Guards against recursion if anything we call is itself hooked. */
static __thread int in_tracer;

/* ------------------------------------------------------------- json buffer */

struct line {
    char buf[LINE_CAP];
    size_t len;
    int overflow;
};

static void ln_raw(struct line *l, const char *s, size_t n)
{
    if (l->overflow)
        return;
    if (n > sizeof(l->buf) - l->len - 1) {
        l->overflow = 1;
        return;
    }
    memcpy(l->buf + l->len, s, n);
    l->len += n;
}

static void ln_str(struct line *l, const char *s)
{
    ln_raw(l, s, strlen(s));
}

/* Emit a JSON string value with the escaping the JSON spec requires. */
static void ln_json_string(struct line *l, const char *s)
{
    ln_str(l, "\"");
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char esc[8];
        switch (*p) {
        case '"':  ln_str(l, "\\\""); break;
        case '\\': ln_str(l, "\\\\"); break;
        case '\n': ln_str(l, "\\n"); break;
        case '\r': ln_str(l, "\\r"); break;
        case '\t': ln_str(l, "\\t"); break;
        default:
            if (*p < 0x20 || *p == 0x7f) {
                snprintf(esc, sizeof(esc), "\\u%04x", *p);
                ln_str(l, esc);
            } else {
                ln_raw(l, (const char *)p, 1);
            }
        }
    }
    ln_str(l, "\"");
}

static void ln_kv_str(struct line *l, const char *key, const char *value)
{
    ln_str(l, ",\"");
    ln_str(l, key);
    ln_str(l, "\":");
    ln_json_string(l, value);
}

static void ln_kv_ll(struct line *l, const char *key, long long value)
{
    char tmp[48];
    snprintf(tmp, sizeof(tmp), ",\"%s\":%lld", key, value);
    ln_str(l, tmp);
}

static void ln_kv_ull(struct line *l, const char *key, unsigned long long value)
{
    char tmp[48];
    snprintf(tmp, sizeof(tmp), ",\"%s\":%llu", key, value);
    ln_str(l, tmp);
}

static void ln_kv_hex(struct line *l, const char *key, unsigned long long value)
{
    char tmp[48];
    snprintf(tmp, sizeof(tmp), ",\"%s\":\"0x%llx\"", key, value);
    ln_str(l, tmp);
}

static void ln_kv_bool(struct line *l, const char *key, int value)
{
    ln_str(l, ",\"");
    ln_str(l, key);
    ln_str(l, "\":");
    ln_str(l, value ? "true" : "false");
}

static void ln_kv_hexbytes(struct line *l, const char *key,
                           const unsigned char *data, size_t n)
{
    static const char digits[] = "0123456789abcdef";
    ln_str(l, ",\"");
    ln_str(l, key);
    ln_str(l, "\":\"");
    for (size_t i = 0; i < n; i++) {
        char pair[2] = { digits[data[i] >> 4], digits[data[i] & 0xf] };
        ln_raw(l, pair, 2);
    }
    ln_str(l, "\"");
}

/* --------------------------------------------------------------- helpers */

static long long now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return -1;
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static long long now_realtime_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
        return -1;
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static long tracer_tid(void)
{
    return (long)syscall(SYS_gettid);
}

/*
 * Copy up to n bytes of the traced process's own memory without risking a
 * fault. process_vm_readv() against our own pid returns EFAULT for an
 * unmapped source instead of delivering SIGSEGV, which keeps rule 1 (never
 * alter the traced process) true even when a buffer pointer is stale.
 */
static ssize_t safe_copy(void *dst, const void *src, size_t n)
{
    if (src == NULL || n == 0)
        return 0;
    struct iovec local = { .iov_base = dst, .iov_len = n };
    struct iovec remote = { .iov_base = (void *)src, .iov_len = n };
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
}

/*
 * Read an unsigned limit from the environment and clamp it into [floor, cap].
 * ``*requested`` receives the value the environment actually asked for (the
 * default when the variable is absent or unparseable), so the header can state
 * both what was asked for and what is in force.
 */
static unsigned long parse_env_uint(const char *name, unsigned long dflt,
                                    unsigned long floor, unsigned long cap,
                                    unsigned long long *requested)
{
    unsigned long long value = dflt;
    const char *raw = getenv(name);
    if (raw != NULL && *raw != '\0') {
        char *end = NULL;
        errno = 0;
        unsigned long long parsed = strtoull(raw, &end, 10);
        if (errno == 0 && end != raw && *end == '\0')
            value = parsed;
    }
    *requested = value;
    if (value > cap)
        return cap;
    if (value < floor)
        return floor;
    return (unsigned long)value;
}

/* -------------------------------------------------------------- emission */

/* Defined below; emit_locked() creates the output file on first use. */
static int ensure_output_locked(void);

/*
 * Append one assembled line to the output. Called with out_lock held.
 * A line that overflowed its buffer, or that would push the file past its
 * byte cap, is counted and announced rather than silently dropped.
 */
/*
 * write() may store fewer bytes than asked for, and a short write in the middle
 * of a JSON line would silently corrupt every record after it. So every write
 * loops until the whole buffer is out, retrying EINTR. Returns the number of
 * bytes actually written, which is less than n only when the file could not
 * take them.
 */
static size_t write_all(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    size_t done = 0;
    while (done < n) {
        ssize_t w = write(fd, p + done, n - done);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (w == 0)
            break;
        done += (size_t)w;
    }
    return done;
}

/*
 * Say, once, that the byte cap stopped the capture. Called with out_lock held
 * and only from emit_locked(), which has already refused an event that would
 * intrude on LIMIT_RESERVE - so this record always has room.
 */
static void announce_limit_locked(void)
{
    if (limit_announced)
        return;
    limit_announced = 1;
    const char *note =
        "{\"op\":\"limit\",\"reason\":\"max_output_bytes\","
        "\"note\":\"capture stopped at the byte cap; later calls are absent, "
        "not idle\"}\n";
    size_t n = strlen(note);
    if (written_bytes + n > max_bytes)
        return;
    written_bytes += write_all(out_fd, note, n);
}

/*
 * ``reserved`` marks the two records that explain a truncated capture - the
 * ``limit`` record and the closing ``summary``. They may spend the reserve;
 * ordinary events may not, which is what guarantees the reserve is still there
 * when they are written.
 */
static void emit_locked_ex(struct line *l, int reserved)
{
    if (!ensure_output_locked())
        return;
    if (l->overflow) {
        dropped_events++;
        return;
    }
    unsigned long budget = reserved ? max_bytes : max_bytes - LIMIT_RESERVE;
    if (written_bytes + l->len + 1 > budget) {
        if (!reserved)
            announce_limit_locked();
        dropped_events++;
        return;
    }
    l->buf[l->len++] = '\n';
    size_t w = write_all(out_fd, l->buf, l->len);
    written_bytes += (unsigned long)w;
    if (w < l->len) {
        /* The file now ends inside a record. Count the loss, and try for a
         * newline so the damage is one unparseable line the decoder reports as
         * a problem rather than a corrupt remainder. */
        dropped_events++;
        if (w > 0)
            written_bytes += (unsigned long)write_all(out_fd, "\n", 1);
    }
}

static void emit_locked(struct line *l)
{
    emit_locked_ex(l, 0);
}

/* Start a line with the fields every event carries. Caller holds out_lock. */
static void begin_event(struct line *l, const char *op, long long t_ns)
{
    l->len = 0;
    l->overflow = 0;
    char head[128];
    snprintf(head, sizeof(head), "{\"seq\":%llu,\"t_ns\":%lld,\"pid\":%ld,\"tid\":%ld",
             ++seq_counter, t_ns, (long)getpid(), tracer_tid());
    ln_str(l, head);
    ln_kv_str(l, "op", op);
}

static void end_event(struct line *l)
{
    ln_str(l, "}");
    emit_locked(l);
}

/* End an event that is allowed to spend the reserve; see emit_locked_ex(). */
static void end_event_reserved(struct line *l)
{
    ln_str(l, "}");
    emit_locked_ex(l, 1);
}

/* --------------------------------------------------------- fd bookkeeping */

static int is_tracked(int fd)
{
    if (!enabled || fd < 0 || fd >= FD_TABLE_SIZE)
        return 0;
    return tracked_fd[fd];
}

static void note_fd_overflow_locked(int fd)
{
    if (fd_overflow_announced)
        return;
    fd_overflow_announced = 1;
    struct line l;
    begin_event(&l, "limit", now_ns());
    ln_kv_str(&l, "reason", "fd_out_of_table");
    ln_kv_ll(&l, "fd", fd);
    ln_kv_str(&l, "note",
              "a matching fd was above the tracer's fd table size; "
              "its transfers are absent from this capture");
    end_event(&l);
}

/*
 * Decide whether a freshly opened fd is the one node we are allowed to record.
 * Both tests must pass: the canonical path must equal the canonical target,
 * and the fd's st_rdev must equal the target's. Requiring both for every newly
 * observed descriptor is what keeps the capture scoped to the single node
 * resolved at startup - another usbfs node open at the same time is refused on
 * one half or the other.
 *
 * This scopes a *snapshot* of a node; it does not authenticate physical
 * identity across a hotplug. Linux may reuse the same bus/device number, hence
 * the same canonical path and the same st_rdev, for whatever enumerates next,
 * so path plus rdev cannot prove the same physical device stayed there. The
 * operator resolves the node from the device serial immediately before the
 * bounded run and treats a disconnect or re-enumeration as invalidating the
 * target; re-establishing identity from inside the tracer would require USB
 * I/O of its own, which this file does not do.
 */
static void consider_fd(int fd, const char *path)
{
    if (!enabled || fd < 0 || path == NULL)
        return;

    char resolved[PATH_MAX];
    if (realpath(path, resolved) == NULL)
        return;
    if (strcmp(resolved, target_path) != 0)
        return;

    struct stat st;
    if (fstat(fd, &st) != 0)
        return;
    if (!S_ISCHR(st.st_mode) || st.st_rdev != target_rdev)
        return;

    long long t = now_ns();
    pthread_mutex_lock(&out_lock);
    if (fd >= FD_TABLE_SIZE) {
        note_fd_overflow_locked(fd);
        pthread_mutex_unlock(&out_lock);
        return;
    }
    tracked_fd[fd] = 1;
    struct line l;
    begin_event(&l, "open", t);
    ln_kv_ll(&l, "fd", fd);
    ln_kv_str(&l, "path", resolved);
    end_event(&l);
    pthread_mutex_unlock(&out_lock);
}

/*
 * Last-resort identification: does this descriptor point at the target device?
 *
 * Hooking the open family is not by itself sufficient, and that is a measured
 * fact rather than a precaution. The vendor SDK imports ``__open_2`` - the
 * _FORTIFY_SOURCE rewrite of ``open()`` with compile-time-constant flags - so a
 * tracer that hooked only ``open``/``openat`` saw the device opened by a symbol
 * it did not interpose and recorded nothing at all.
 *
 * Rather than chase every spelling, an untracked descriptor that receives a
 * usbfs ioctl is checked directly, against *both* halves of the identity test
 * consider_fd() applies: the descriptor's own path, read back from
 * /proc/self/fd, must equal the canonical target path, and its st_rdev must
 * equal the target's. Requiring both means adoption is neither a weaker nor a
 * stronger door into the capture than an intercepted open: it applies exactly
 * the same test, with exactly the same limitation - it scopes the capture to
 * the node snapshot resolved at startup and cannot prove physical identity
 * across a hotplug - while it holds however the descriptor came to exist: a
 * raw syscall, an inherited fd, or a spelling of open() that does not exist
 * yet.
 *
 * It also fails closed. If /proc is not mounted the link cannot be read, so no
 * descriptor is ever adopted and the capture falls back to whatever the open
 * hooks saw; nothing unverified is recorded.
 *
 * readlink() and fstat() read path and inode metadata; neither issues a USB
 * operation.
 */
static int fd_path_is_target(int fd)
{
    char link[64];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    char resolved[PATH_MAX];
    ssize_t n = readlink(link, resolved, sizeof(resolved) - 1);
    if (n < 0 || (size_t)n >= sizeof(resolved) - 1)
        return 0;
    resolved[n] = '\0';
    return strcmp(resolved, target_path) == 0;
}

static int adopt_if_target(int fd)
{
    struct stat st;
    if (fstat(fd, &st) != 0)
        return 0;
    if (!S_ISCHR(st.st_mode) || st.st_rdev != target_rdev)
        return 0;
    if (!fd_path_is_target(fd))
        return 0;

    long long t = now_ns();
    pthread_mutex_lock(&out_lock);
    if (fd >= FD_TABLE_SIZE) {
        note_fd_overflow_locked(fd);
        pthread_mutex_unlock(&out_lock);
        return 0;
    }
    if (!tracked_fd[fd]) {
        tracked_fd[fd] = 1;
        struct line l;
        begin_event(&l, "adopt", t);
        ln_kv_ll(&l, "fd", fd);
        ln_kv_str(&l, "path", target_path);
        ln_kv_str(&l, "note",
                  "descriptor identified by its /proc/self/fd path and st_rdev "
                  "on its first usbfs ioctl, not by an intercepted open");
        end_event(&l);
    }
    pthread_mutex_unlock(&out_lock);
    return 1;
}

static void untrack_fd(int fd, const char *op)
{
    if (!enabled || fd < 0 || fd >= FD_TABLE_SIZE)
        return;
    long long t = now_ns();
    pthread_mutex_lock(&out_lock);
    if (tracked_fd[fd]) {
        tracked_fd[fd] = 0;
        struct line l;
        begin_event(&l, op, t);
        ln_kv_ll(&l, "fd", fd);
        end_event(&l);
    }
    pthread_mutex_unlock(&out_lock);
}

/* ------------------------------------------------------- ioctl decoding */

static const char *ioctl_name(unsigned long request)
{
    switch (request) {
    case USBDEVFS_CONTROL: return "USBDEVFS_CONTROL";
    case USBDEVFS_BULK: return "USBDEVFS_BULK";
    case USBDEVFS_RESETEP: return "USBDEVFS_RESETEP";
    case USBDEVFS_SETINTERFACE: return "USBDEVFS_SETINTERFACE";
    case USBDEVFS_SETCONFIGURATION: return "USBDEVFS_SETCONFIGURATION";
    case USBDEVFS_GETDRIVER: return "USBDEVFS_GETDRIVER";
    case USBDEVFS_SUBMITURB: return "USBDEVFS_SUBMITURB";
    case USBDEVFS_DISCARDURB: return "USBDEVFS_DISCARDURB";
    case USBDEVFS_REAPURB: return "USBDEVFS_REAPURB";
    case USBDEVFS_REAPURBNDELAY: return "USBDEVFS_REAPURBNDELAY";
    case USBDEVFS_DISCSIGNAL: return "USBDEVFS_DISCSIGNAL";
    case USBDEVFS_CLAIMINTERFACE: return "USBDEVFS_CLAIMINTERFACE";
    case USBDEVFS_RELEASEINTERFACE: return "USBDEVFS_RELEASEINTERFACE";
    case USBDEVFS_CONNECTINFO: return "USBDEVFS_CONNECTINFO";
    case USBDEVFS_IOCTL: return "USBDEVFS_IOCTL";
    case USBDEVFS_HUB_PORTINFO: return "USBDEVFS_HUB_PORTINFO";
    case USBDEVFS_RESET: return "USBDEVFS_RESET";
    case USBDEVFS_CLEAR_HALT: return "USBDEVFS_CLEAR_HALT";
    case USBDEVFS_DISCONNECT: return "USBDEVFS_DISCONNECT";
    case USBDEVFS_CONNECT: return "USBDEVFS_CONNECT";
    case USBDEVFS_CLAIM_PORT: return "USBDEVFS_CLAIM_PORT";
    case USBDEVFS_RELEASE_PORT: return "USBDEVFS_RELEASE_PORT";
    case USBDEVFS_GET_CAPABILITIES: return "USBDEVFS_GET_CAPABILITIES";
    case USBDEVFS_DISCONNECT_CLAIM: return "USBDEVFS_DISCONNECT_CLAIM";
#ifdef USBDEVFS_ALLOC_STREAMS
    case USBDEVFS_ALLOC_STREAMS: return "USBDEVFS_ALLOC_STREAMS";
    case USBDEVFS_FREE_STREAMS: return "USBDEVFS_FREE_STREAMS";
#endif
#ifdef USBDEVFS_DROP_PRIVILEGES
    case USBDEVFS_DROP_PRIVILEGES: return "USBDEVFS_DROP_PRIVILEGES";
#endif
#ifdef USBDEVFS_GET_SPEED
    case USBDEVFS_GET_SPEED: return "USBDEVFS_GET_SPEED";
#endif
#ifdef USBDEVFS_FORBID_SUSPEND
    case USBDEVFS_FORBID_SUSPEND: return "USBDEVFS_FORBID_SUSPEND";
    case USBDEVFS_ALLOW_SUSPEND: return "USBDEVFS_ALLOW_SUSPEND";
#endif
    default: return NULL;
    }
}

static const char *urb_type_name(unsigned char type)
{
    switch (type) {
    case USBDEVFS_URB_TYPE_ISO: return "iso";
    case USBDEVFS_URB_TYPE_INTERRUPT: return "interrupt";
    case USBDEVFS_URB_TYPE_CONTROL: return "control";
    case USBDEVFS_URB_TYPE_BULK: return "bulk";
    default: return "unknown";
    }
}

static void ln_endpoint(struct line *l, unsigned char endpoint)
{
    ln_kv_ull(l, "endpoint", endpoint);
    ln_kv_ull(l, "endpoint_number", endpoint & 0x0fu);
    ln_kv_str(l, "direction", (endpoint & 0x80u) ? "in" : "out");
}

/* Decode the 8-byte USB control setup packet that prefixes a control buffer. */
static void ln_setup(struct line *l, const unsigned char *s)
{
    ln_kv_ull(l, "bmRequestType", s[0]);
    ln_kv_ull(l, "bRequest", s[1]);
    ln_kv_ull(l, "wValue", (unsigned)s[2] | ((unsigned)s[3] << 8));
    ln_kv_ull(l, "wIndex", (unsigned)s[4] | ((unsigned)s[5] << 8));
    ln_kv_ull(l, "wLength", (unsigned)s[6] | ((unsigned)s[7] << 8));
    ln_kv_str(l, "setup_direction", (s[0] & 0x80u) ? "in" : "out");
}

/*
 * Copy at most max_payload bytes from a user buffer into the line under the
 * given key, always recording how many bytes existed so truncation is visible
 * rather than implied.
 */
static void ln_payload(struct line *l, const char *key, const void *buffer,
                       long available)
{
    if (available < 0)
        available = 0;
    unsigned long want = (unsigned long)available;
    int truncated = 0;
    if (want > max_payload) {
        want = max_payload;
        truncated = 1;
    }
    ln_kv_ll(l, "payload_available", available);
    ln_kv_ull(l, "payload_captured", want);
    ln_kv_bool(l, "payload_truncated", truncated);
    if (want == 0) {
        ln_kv_hexbytes(l, key, (const unsigned char *)"", 0);
        return;
    }
    unsigned char tmp[PAYLOAD_CAP];
    ssize_t got = safe_copy(tmp, buffer, want);
    if (got < 0) {
        ln_kv_str(l, "payload_error", "unreadable");
        ln_kv_ull(l, "payload_captured", 0);
        ln_kv_hexbytes(l, key, tmp, 0);
        return;
    }
    ln_kv_hexbytes(l, key, tmp, (size_t)got);
}

/* Record the fields of a struct usbdevfs_urb that we can read safely. */
static void ln_urb(struct line *l, const void *urb_ptr, int with_submit_payload,
                   int with_reap_payload)
{
    struct usbdevfs_urb urb;
    ln_kv_hex(l, "urb", (unsigned long long)(uintptr_t)urb_ptr);
    ssize_t got = safe_copy(&urb, urb_ptr, sizeof(urb));
    if (got != (ssize_t)sizeof(urb)) {
        ln_kv_str(l, "urb_error", "unreadable");
        return;
    }
    ln_kv_ull(l, "urb_type", urb.type);
    ln_kv_str(l, "urb_type_name", urb_type_name(urb.type));
    ln_endpoint(l, urb.endpoint);
    ln_kv_ll(l, "status", urb.status);
    ln_kv_ull(l, "flags", urb.flags);
    ln_kv_ll(l, "buffer_length", urb.buffer_length);
    ln_kv_ll(l, "actual_length", urb.actual_length);
    ln_kv_ll(l, "start_frame", urb.start_frame);
    ln_kv_ll(l, "number_of_packets", urb.number_of_packets);
    ln_kv_ll(l, "error_count", urb.error_count);
    ln_kv_ull(l, "signr", urb.signr);
    ln_kv_hex(l, "usercontext", (unsigned long long)(uintptr_t)urb.usercontext);
    if (urb.type == USBDEVFS_URB_TYPE_ISO)
        ln_kv_str(l, "note", "iso packet descriptors are not decoded");

    const unsigned char *data = (const unsigned char *)urb.buffer;
    long data_len = urb.buffer_length;
    int is_in = (urb.endpoint & 0x80u) != 0;

    if (urb.type == USBDEVFS_URB_TYPE_CONTROL) {
        unsigned char setup[8];
        if (safe_copy(setup, urb.buffer, sizeof(setup)) == (ssize_t)sizeof(setup)) {
            ln_setup(l, setup);
            is_in = (setup[0] & 0x80u) != 0;
        } else {
            ln_kv_str(l, "setup_error", "unreadable");
        }
        /* A control URB's buffer is the 8-byte setup packet then the data. */
        data = data ? data + 8 : NULL;
        data_len = data_len >= 8 ? data_len - 8 : 0;
    }

    /* At submit only an OUT buffer holds meaningful bytes; at reap only an IN
     * buffer does, and then only actual_length of them. */
    if (with_submit_payload && !is_in)
        ln_payload(l, "data", data, data_len);
    else if (with_reap_payload && is_in) {
        long got = urb.actual_length;
        if (urb.type == USBDEVFS_URB_TYPE_CONTROL && got >= 0 && got > data_len)
            got = data_len;
        ln_payload(l, "data", data, got);
    }
}

/* Decode the small fixed-shape argument structs of the non-URB ioctls. */
static void ln_simple_arg(struct line *l, unsigned long request, void *arg)
{
    switch (request) {
    case USBDEVFS_CLAIMINTERFACE:
    case USBDEVFS_RELEASEINTERFACE: {
        unsigned int ifno = 0;
        if (safe_copy(&ifno, arg, sizeof(ifno)) == (ssize_t)sizeof(ifno))
            ln_kv_ull(l, "interface", ifno);
        break;
    }
    case USBDEVFS_CLEAR_HALT:
    case USBDEVFS_RESETEP: {
        unsigned int ep = 0;
        if (safe_copy(&ep, arg, sizeof(ep)) == (ssize_t)sizeof(ep))
            ln_endpoint(l, (unsigned char)ep);
        break;
    }
    case USBDEVFS_SETINTERFACE: {
        struct usbdevfs_setinterface si;
        if (safe_copy(&si, arg, sizeof(si)) == (ssize_t)sizeof(si)) {
            ln_kv_ull(l, "interface", si.interface);
            ln_kv_ull(l, "altsetting", si.altsetting);
        }
        break;
    }
    case USBDEVFS_GETDRIVER: {
        struct usbdevfs_getdriver gd;
        if (safe_copy(&gd, arg, sizeof(gd)) == (ssize_t)sizeof(gd)) {
            gd.driver[sizeof(gd.driver) - 1] = '\0';
            ln_kv_ull(l, "interface", gd.interface);
            ln_kv_str(l, "driver", gd.driver);
        }
        break;
    }
    case USBDEVFS_IOCTL: {
        /* The nested-ioctl passthrough; this is how libusb detaches and
         * reattaches a kernel driver on one interface. */
        struct usbdevfs_ioctl ui;
        if (safe_copy(&ui, arg, sizeof(ui)) == (ssize_t)sizeof(ui)) {
            ln_kv_ll(l, "interface", ui.ifno);
            ln_kv_hex(l, "inner_request", (unsigned long long)(unsigned int)ui.ioctl_code);
            const char *inner = ioctl_name((unsigned long)(unsigned int)ui.ioctl_code);
            if (inner != NULL)
                ln_kv_str(l, "inner_name", inner);
        }
        break;
    }
    case USBDEVFS_DISCONNECT_CLAIM: {
        struct usbdevfs_disconnect_claim dc;
        if (safe_copy(&dc, arg, sizeof(dc)) == (ssize_t)sizeof(dc)) {
            dc.driver[sizeof(dc.driver) - 1] = '\0';
            ln_kv_ull(l, "interface", dc.interface);
            ln_kv_ull(l, "claim_flags", dc.flags);
            ln_kv_str(l, "driver", dc.driver);
        }
        break;
    }
    case USBDEVFS_GET_CAPABILITIES: {
        unsigned int caps = 0;
        if (safe_copy(&caps, arg, sizeof(caps)) == (ssize_t)sizeof(caps))
            ln_kv_hex(l, "capabilities", caps);
        break;
    }
    case USBDEVFS_CONTROL: {
        struct usbdevfs_ctrltransfer ct;
        if (safe_copy(&ct, arg, sizeof(ct)) == (ssize_t)sizeof(ct)) {
            unsigned char setup[8] = {
                ct.bRequestType, ct.bRequest,
                (unsigned char)(ct.wValue & 0xff), (unsigned char)(ct.wValue >> 8),
                (unsigned char)(ct.wIndex & 0xff), (unsigned char)(ct.wIndex >> 8),
                (unsigned char)(ct.wLength & 0xff), (unsigned char)(ct.wLength >> 8),
            };
            ln_setup(l, setup);
            ln_kv_ull(l, "timeout_ms", ct.timeout);
            ln_payload(l, "data", ct.data, ct.wLength);
        }
        break;
    }
    case USBDEVFS_BULK: {
        struct usbdevfs_bulktransfer bt;
        if (safe_copy(&bt, arg, sizeof(bt)) == (ssize_t)sizeof(bt)) {
            ln_endpoint(l, (unsigned char)bt.ep);
            ln_kv_ull(l, "timeout_ms", bt.timeout);
            ln_payload(l, "data", bt.data, bt.len);
        }
        break;
    }
    default:
        break;
    }
}

/* strerrorname_np() is glibc >= 2.32; older libcs get the numeric code only. */
static const char *errno_name(int err)
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32))
    const char *name = strerrorname_np(err);
    return name != NULL ? name : "UNKNOWN";
#else
    (void)err;
    return "UNKNOWN";
#endif
}

/* ------------------------------------------------------------------- init */

static void disable(const char *reason)
{
    enabled = 0;
    fprintf(stderr, "[usbfs_trace] capture disabled: %s\n", reason);
}

static void write_header(void)
{
    struct line l;
    l.len = 0;
    l.overflow = 0;
    char head[160];
    snprintf(head, sizeof(head), "{\"op\":\"header\",\"format\":\"%s\",\"version\":%d",
             TRACE_FORMAT, TRACE_VERSION);
    ln_str(&l, head);
    ln_kv_str(&l, "target", target_path);
    ln_kv_ull(&l, "target_rdev_major", (unsigned long long)major(target_rdev));
    ln_kv_ull(&l, "target_rdev_minor", (unsigned long long)minor(target_rdev));
    ln_kv_ll(&l, "pid", (long long)getpid());
    ln_kv_ll(&l, "ppid", (long long)getppid());
    ln_kv_str(&l, "path", out_path);
    ln_kv_ull(&l, "max_payload_bytes", max_payload);
    ln_kv_ull(&l, "max_output_bytes", max_bytes);
    /* What the environment asked for, next to what is in force. When the two
     * differ the caller's request was clamped, and the evidence says so. */
    ln_kv_ull(&l, "requested_max_payload_bytes", requested_payload);
    ln_kv_ull(&l, "requested_max_output_bytes", requested_bytes);
    ln_kv_bool(&l, "limits_clamped",
               requested_payload != (unsigned long long)max_payload ||
                   requested_bytes != (unsigned long long)max_bytes);
    ln_kv_ull(&l, "limit_reserve_bytes", LIMIT_RESERVE);
    ln_kv_str(&l, "clock", "CLOCK_MONOTONIC");
    ln_kv_ll(&l, "started_realtime_ns", now_realtime_ns());
    ln_str(&l, ",\"limitations\":[");
    ln_json_string(&l, "seq is emission order (assigned after the call returns); "
                       "t_ns is call-entry time on CLOCK_MONOTONIC");
    ln_str(&l, ",");
    ln_json_string(&l, "payload bytes are capped per event by max_payload_bytes; "
                       "payload_available records the untruncated length");
    ln_str(&l, ",");
    ln_json_string(&l, "isochronous per-packet descriptors are not decoded");
    ln_str(&l, ",");
    ln_json_string(&l, "only ioctls on file descriptors whose /proc/self/fd link "
                       "equals the target path and whose st_rdev matches are recorded");
    ln_str(&l, ",");
    ln_json_string(&l, "the last limit_reserve_bytes of max_output_bytes are held "
                       "back for the limit and summary records, so events stop "
                       "short of the cap and a capped capture still states its "
                       "own losses");
    ln_str(&l, ",");
    ln_json_string(&l, "URBs still in flight when the process exits have no reap event");
    ln_str(&l, ",");
    ln_json_string(&l, "traffic from other processes and from the kernel's own "
                       "uvcvideo/usbhid drivers is out of reach of this mechanism");
    ln_str(&l, ",");
    ln_json_string(&l, "the closing summary record is written at this library's "
                       "teardown; a library torn down later can still append events");
    ln_str(&l, "]");
    ln_str(&l, "}");
    emit_locked(&l);
}

/*
 * Create this process's own output file, on first use rather than at load.
 *
 * Two facts force this shape. The traced program is free to fork and exec -
 * even a bare ``timeout 15 probe ...`` wrapper means two images run with this
 * library loaded - and every image re-runs the constructor. And a fork child
 * that is about to exec must not claim a filename the exec'd image will need.
 * So the name carries the pid (one file per process, never an interleave), and
 * nothing is created until a process actually records something. A process
 * that never touches the target device therefore leaves no file at all, which
 * is the truthful result rather than an empty one.
 *
 * Called with out_lock held.
 */
static int ensure_output_locked(void)
{
    if (out_fd >= 0)
        return 1;
    if (!enabled || output_failed)
        return 0;

    /* The precision bound is what lets the compiler see the result fits;
     * tracer_init() has already rejected a prefix long enough to need it. */
    snprintf(out_path, sizeof(out_path), "%.*s.%ld",
             (int)(sizeof(out_path) - 24), out_prefix, (long)getpid());
    /* O_EXCL: never overwrite an existing capture. O_NOFOLLOW: never follow a
     * symlink someone else planted at the output path. */
    int fd = real_open(out_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        output_failed = 1;
        disable("output file could not be created exclusively");
        return 0;
    }
    out_fd = fd;
    write_header();
    return 1;
}

#define RESOLVE(slot, symbol)                          \
    do {                                               \
        void *sym = dlsym(RTLD_NEXT, (symbol));        \
        memcpy(&(slot), &sym, sizeof(sym));            \
    } while (0)

static void tracer_init(void)
{
    /* ISO C has no conversion between void * and a function pointer, so the
     * result is moved through the object representation instead of assigned. */
    RESOLVE(real_ioctl, "ioctl");
    RESOLVE(real_open, "open");
    RESOLVE(real_open64, "open64");
    RESOLVE(real_openat, "openat");
    RESOLVE(real_openat64, "openat64");
    RESOLVE(real___open_2, "__open_2");
    RESOLVE(real___open64_2, "__open64_2");
    RESOLVE(real___openat_2, "__openat_2");
    RESOLVE(real___openat64_2, "__openat64_2");
    RESOLVE(real_close, "close");
    RESOLVE(real_dup, "dup");
    RESOLVE(real_dup2, "dup2");
    RESOLVE(real_dup3, "dup3");

    const char *target = getenv(ENV_TARGET);
    const char *out = getenv(ENV_OUT);
    if (target == NULL || *target == '\0' || out == NULL || *out == '\0') {
        /* Not opted in. Stay silent: the library is a no-op passthrough. */
        enabled = 0;
        return;
    }

    enabled = 1;
    /* A payload cap of zero is a legitimate request (record shapes, no bytes),
     * so payloads have no floor. The output cap does have one: a file too small
     * to hold the header plus the reserved ``limit`` record could not be read
     * back at all. Both clamps are reported in the header. */
    max_payload = (unsigned)parse_env_uint(ENV_MAX_PAYLOAD, PAYLOAD_DEFAULT, 0,
                                           PAYLOAD_CAP, &requested_payload);
    max_bytes = parse_env_uint(ENV_MAX_BYTES, OUTPUT_DEFAULT, OUTPUT_MIN,
                               OUTPUT_CAP, &requested_bytes);

    if (realpath(target, target_path) == NULL) {
        disable("target path does not resolve");
        return;
    }
    if (strncmp(target_path, "/dev/bus/usb/", 13) != 0) {
        disable("target is not under /dev/bus/usb/");
        return;
    }
    struct stat st;
    if (stat(target_path, &st) != 0 || !S_ISCHR(st.st_mode)) {
        disable("target is not a character device");
        return;
    }
    target_rdev = st.st_rdev;

    if (strlen(out) + 24 >= sizeof(out_prefix)) {
        disable("output path is too long");
        return;
    }
    snprintf(out_prefix, sizeof(out_prefix), "%s", out);
    /* The file itself is created lazily by ensure_output_locked(). */
}

/*
 * A fork with no exec inherits our descriptor and our counters, so the child
 * would append its records to the parent's file under the parent's sequence
 * numbers. These handlers give the child a file of its own instead. The
 * prepare/parent pair exists so the child never inherits a held mutex.
 */
static void atfork_prepare(void)
{
    if (enabled)
        pthread_mutex_lock(&out_lock);
}

static void atfork_parent(void)
{
    if (enabled)
        pthread_mutex_unlock(&out_lock);
}

static void atfork_child(void)
{
    if (!enabled) {
        return;
    }
    pthread_mutex_unlock(&out_lock);
    if (out_fd >= 0)
        real_close(out_fd);
    out_fd = -1;
    seq_counter = 0;
    written_bytes = 0;
    dropped_events = 0;
    limit_announced = 0;
    fd_overflow_announced = 0;
    output_failed = 0;
    /* Inherited descriptors still refer to the same device, so tracked_fd is
     * deliberately kept as it is. The child's own file, if it ever records
     * anything, is created lazily under the child's pid. */
}

__attribute__((constructor)) static void tracer_ctor(void)
{
    pthread_once(&init_once, tracer_init);
    if (enabled)
        pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
}

__attribute__((destructor)) static void tracer_dtor(void)
{
    if (!enabled || out_fd < 0)
        return;
    pthread_mutex_lock(&out_lock);
    struct line l;
    begin_event(&l, "summary", now_ns());
    ln_kv_ull(&l, "events_written", seq_counter);
    ln_kv_ull(&l, "bytes_written", written_bytes);
    ln_kv_ull(&l, "events_dropped", dropped_events);
    ln_kv_str(&l, "note",
              "written when this library is torn down, which is not necessarily "
              "last: a library unloaded after this one can still record events, "
              "so records may follow. Trust the decoder's own counts.");
    end_event_reserved(&l);
    pthread_mutex_unlock(&out_lock);
    /* The output fd is deliberately left for process teardown to close; the
     * traced process's own cleanup order is not ours to change. */
}

/* --------------------------------------------------------------- the hooks
 *
 * Every hook has the same shape, and the shape is the transparency guarantee:
 * record the caller's errno on entry, forward to the real libc symbol with the
 * caller's arguments and the caller's errno, then restore the errno the real
 * call produced before returning. The tracer's own bookkeeping - tracer_init(),
 * fstat(), readlink(), clock_gettime(), write() - sets errno freely in between,
 * and none of that is allowed to reach the traced program.
 */

int ioctl(int fd, unsigned long request, ...)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);

    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    if (in_tracer) {
        errno = entry_errno;
        return real_ioctl(fd, request, arg);
    }
    if (!is_tracked(fd)) {
        /* Only usbfs requests are worth a stat; everything else passes
         * straight through with no added syscall at all. */
        if (!enabled || _IOC_TYPE(request) != 'U' || fd < 0) {
            errno = entry_errno;
            return real_ioctl(fd, request, arg);
        }
        in_tracer = 1;
        int adopted = adopt_if_target(fd);
        in_tracer = 0;
        if (!adopted) {
            errno = entry_errno;
            return real_ioctl(fd, request, arg);
        }
    }

    long long t = now_ns();
    errno = entry_errno;
    int ret = real_ioctl(fd, request, arg);
    int saved_errno = errno;

    in_tracer = 1;
    long long t_end = now_ns();
    const char *name = ioctl_name(request);

    pthread_mutex_lock(&out_lock);
    struct line l;
    begin_event(&l, "ioctl", t);
    ln_kv_ll(&l, "fd", fd);
    ln_kv_hex(&l, "request", (unsigned long long)request);
    if (name != NULL)
        ln_kv_str(&l, "name", name);
    ln_kv_ll(&l, "ret", ret);
    ln_kv_ll(&l, "errno", ret < 0 ? saved_errno : 0);
    if (ret < 0)
        ln_kv_str(&l, "errno_name", errno_name(saved_errno));
    ln_kv_ll(&l, "duration_ns", t_end - t);

    switch (request) {
    case USBDEVFS_SUBMITURB:
        ln_kv_str(&l, "phase", "submit");
        if (ret == 0)
            ln_urb(&l, arg, 1, 0);
        else
            ln_kv_hex(&l, "urb", (unsigned long long)(uintptr_t)arg);
        break;
    case USBDEVFS_DISCARDURB:
        ln_kv_str(&l, "phase", "discard");
        ln_kv_hex(&l, "urb", (unsigned long long)(uintptr_t)arg);
        break;
    case USBDEVFS_REAPURB:
    case USBDEVFS_REAPURBNDELAY: {
        ln_kv_str(&l, "phase", "reap");
        void *reaped = NULL;
        if (ret == 0 && safe_copy(&reaped, arg, sizeof(reaped)) == (ssize_t)sizeof(reaped))
            ln_urb(&l, reaped, 0, 1);
        break;
    }
    default:
        ln_simple_arg(&l, request, arg);
        break;
    }
    end_event(&l);
    pthread_mutex_unlock(&out_lock);
    in_tracer = 0;

    errno = saved_errno;
    return ret;
}

static int open_mode(int flags, va_list ap)
{
    if (flags & (O_CREAT | O_TMPFILE))
        return va_arg(ap, int);
    return 0;
}

int open(const char *path, int flags, ...)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    va_list ap;
    va_start(ap, flags);
    int mode = open_mode(flags, ap);
    va_end(ap);
    errno = entry_errno;
    int fd = real_open(path, flags, mode);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int open64(const char *path, int flags, ...)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    va_list ap;
    va_start(ap, flags);
    int mode = open_mode(flags, ap);
    va_end(ap);
    errno = entry_errno;
    int fd = real_open64 ? real_open64(path, flags, mode)
                         : real_open(path, flags, mode);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int openat(int dirfd, const char *path, int flags, ...)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    va_list ap;
    va_start(ap, flags);
    int mode = open_mode(flags, ap);
    va_end(ap);
    errno = entry_errno;
    int fd = real_openat(dirfd, path, flags, mode);
    int saved = errno;
    /* Only absolute paths are resolved here; a relative openat() against a
     * directory fd is not reconstructed, and would simply not be traced. */
    if (fd >= 0 && !in_tracer && path[0] == '/') {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int openat64(int dirfd, const char *path, int flags, ...)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    va_list ap;
    va_start(ap, flags);
    int mode = open_mode(flags, ap);
    va_end(ap);
    errno = entry_errno;
    int fd = real_openat64 ? real_openat64(dirfd, path, flags, mode)
                           : real_openat(dirfd, path, flags, mode);
    int saved = errno;
    if (fd >= 0 && !in_tracer && path[0] == '/') {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int close(int fd)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    if (!in_tracer && is_tracked(fd)) {
        in_tracer = 1;
        untrack_fd(fd, "close");
        in_tracer = 0;
    }
    errno = entry_errno;
    return real_close(fd);
}

/* dup family: a duplicated descriptor is the same device, so it must inherit
 * tracking, and a dup2/dup3 target must lose any tracking it had. */
static void inherit_tracking(int oldfd, int newfd)
{
    if (!enabled || newfd < 0 || newfd >= FD_TABLE_SIZE)
        return;
    int track = is_tracked(oldfd);
    long long t = now_ns();
    pthread_mutex_lock(&out_lock);
    if (track && !tracked_fd[newfd]) {
        tracked_fd[newfd] = 1;
        struct line l;
        begin_event(&l, "dup", t);
        ln_kv_ll(&l, "fd", newfd);
        ln_kv_ll(&l, "from_fd", oldfd);
        ln_kv_str(&l, "path", target_path);
        end_event(&l);
    } else if (!track && tracked_fd[newfd]) {
        tracked_fd[newfd] = 0;
        struct line l;
        begin_event(&l, "close", t);
        ln_kv_ll(&l, "fd", newfd);
        ln_kv_str(&l, "note", "descriptor replaced by dup2/dup3");
        end_event(&l);
    }
    pthread_mutex_unlock(&out_lock);
}

int dup(int oldfd)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real_dup(oldfd);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        inherit_tracking(oldfd, fd);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int dup2(int oldfd, int newfd)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real_dup2(oldfd, newfd);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        inherit_tracking(oldfd, fd);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int dup3(int oldfd, int newfd, int flags)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real_dup3(oldfd, newfd, flags);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        inherit_tracking(oldfd, fd);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

/*
 * The _FORTIFY_SOURCE spellings of open()/openat(). glibc's headers rewrite a
 * call whose flags are a compile-time constant without O_CREAT into these, so
 * a library built with hardening on never references ``open`` at all. The
 * vendor SDK is one such library.
 */
int __open_2(const char *path, int flags)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real___open_2(path, flags);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int __open64_2(const char *path, int flags)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real___open64_2 ? real___open64_2(path, flags) : real___open_2(path, flags);
    int saved = errno;
    if (fd >= 0 && !in_tracer) {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int __openat_2(int dirfd, const char *path, int flags)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real___openat_2(dirfd, path, flags);
    int saved = errno;
    if (fd >= 0 && !in_tracer && path[0] == '/') {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}

int __openat64_2(int dirfd, const char *path, int flags)
{
    const int entry_errno = errno;
    pthread_once(&init_once, tracer_init);
    errno = entry_errno;
    int fd = real___openat64_2 ? real___openat64_2(dirfd, path, flags)
                               : real___openat_2(dirfd, path, flags);
    int saved = errno;
    if (fd >= 0 && !in_tracer && path[0] == '/') {
        in_tracer = 1;
        consider_fd(fd, path);
        in_tracer = 0;
    }
    errno = saved;
    return fd;
}
