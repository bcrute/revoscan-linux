/*
 * v4l2_layout.c
 *
 * WHY this file exists: the 3d-scanner project talks to V4L2 devices from
 * Python using raw `struct.pack`/`struct.unpack` and `fcntl.ioctl`, which
 * means every field offset, struct size, and ioctl request number baked
 * into the Python code is a hand-transcribed copy of what the C UAPI
 * headers (<linux/videodev2.h>) actually declare on the *target* machine.
 * There is no compiler to catch a mistake in that transcription -- a wrong
 * offset just silently corrupts buffers or misreads capability flags.
 *
 * This program is the ground truth generator: it is compiled and run with
 * the *real* kernel UAPI headers installed on this host, and it prints
 * everything the Python layer needs to cross-check itself against, as
 * JSON, so the numbers can be diffed against the Python constants instead
 * of trusted by inspection.
 *
 * Every value that might not exist on an older kernel header is guarded
 * with #ifdef so this file keeps compiling (and emits JSON `null` for the
 * missing pieces) rather than failing outright on an older distro.
 */

#include <stdio.h>
#include <stddef.h>   /* offsetof */
#include <sys/time.h> /* struct timeval, used inside struct v4l2_buffer */
#include <linux/videodev2.h>

/* asm-generic/ioctl.h defines the _IOC_* encoding macros; pull it in
 * directly (rather than relying on it being dragged in transitively) so
 * we can also report the raw encoding parameters in step 5. */
#include <asm/ioctl.h>

/* ---- small JSON-printing helpers -----------------------------------
 * Kept deliberately dumb (no escaping beyond what we need) because every
 * string we print is a literal we chose, not external input.
 */

/* Track whether the current JSON object/array needs a leading comma. */
static void comma(int *first) {
    if (!*first) {
        printf(",\n");
    } else {
        *first = 0;
    }
}

static void print_size_t(const char *name, size_t value, int *first) {
    comma(first);
    printf("    \"%s\": %zu", name, value);
}

/* ---------------------------------------------------------------------
 * Print sizeof for a curated list of structs used by the buffer/ioctl
 * path in the Python implementation.
 */
static void print_sizeofs(void) {
    int first = 1;
    printf("  \"sizeof\": {\n");
    print_size_t("struct v4l2_capability", sizeof(struct v4l2_capability), &first);
    print_size_t("struct v4l2_format", sizeof(struct v4l2_format), &first);
    print_size_t("struct v4l2_pix_format", sizeof(struct v4l2_pix_format), &first);
    print_size_t("struct v4l2_requestbuffers", sizeof(struct v4l2_requestbuffers), &first);
    print_size_t("struct v4l2_buffer", sizeof(struct v4l2_buffer), &first);
    print_size_t("struct v4l2_queryctrl", sizeof(struct v4l2_queryctrl), &first);
    print_size_t("struct v4l2_control", sizeof(struct v4l2_control), &first);
    print_size_t("struct v4l2_timecode", sizeof(struct v4l2_timecode), &first);
    print_size_t("struct timeval", sizeof(struct timeval), &first);
#ifdef V4L2_MEMORY_MMAP /* proxy: confirms videodev2.h defines the mplane bits we need */
    print_size_t("struct v4l2_plane", sizeof(struct v4l2_plane), &first);
#endif
    printf("\n  },\n");
}

/* ---------------------------------------------------------------------
 * offsetof for every field the Python struct-packing code cares about.
 * Anonymous unions (e.g. v4l2_pix_format's ycbcr_enc/hsv_enc, or
 * v4l2_buffer's request_fd/reserved) promote their members into the
 * enclosing struct's namespace, so offsetof(EnclosingStruct, member)
 * works directly without naming the union.
 */
static void print_offsets(void) {
    int first = 1;
    printf("  \"offsetof\": {\n");

    /* struct v4l2_capability */
    print_size_t("v4l2_capability.driver", offsetof(struct v4l2_capability, driver), &first);
    print_size_t("v4l2_capability.card", offsetof(struct v4l2_capability, card), &first);
    print_size_t("v4l2_capability.bus_info", offsetof(struct v4l2_capability, bus_info), &first);
    print_size_t("v4l2_capability.version", offsetof(struct v4l2_capability, version), &first);
    print_size_t("v4l2_capability.capabilities", offsetof(struct v4l2_capability, capabilities), &first);
#ifdef V4L2_CAP_DEVICE_CAPS /* device_caps was added alongside this capability bit */
    print_size_t("v4l2_capability.device_caps", offsetof(struct v4l2_capability, device_caps), &first);
#endif
    print_size_t("v4l2_capability.reserved", offsetof(struct v4l2_capability, reserved), &first);

    /* struct v4l2_pix_format */
    print_size_t("v4l2_pix_format.width", offsetof(struct v4l2_pix_format, width), &first);
    print_size_t("v4l2_pix_format.height", offsetof(struct v4l2_pix_format, height), &first);
    print_size_t("v4l2_pix_format.pixelformat", offsetof(struct v4l2_pix_format, pixelformat), &first);
    print_size_t("v4l2_pix_format.field", offsetof(struct v4l2_pix_format, field), &first);
    print_size_t("v4l2_pix_format.bytesperline", offsetof(struct v4l2_pix_format, bytesperline), &first);
    print_size_t("v4l2_pix_format.sizeimage", offsetof(struct v4l2_pix_format, sizeimage), &first);
    print_size_t("v4l2_pix_format.colorspace", offsetof(struct v4l2_pix_format, colorspace), &first);
    print_size_t("v4l2_pix_format.priv", offsetof(struct v4l2_pix_format, priv), &first);
    print_size_t("v4l2_pix_format.flags", offsetof(struct v4l2_pix_format, flags), &first);
    print_size_t("v4l2_pix_format.ycbcr_enc", offsetof(struct v4l2_pix_format, ycbcr_enc), &first);
    print_size_t("v4l2_pix_format.hsv_enc", offsetof(struct v4l2_pix_format, hsv_enc), &first);
    print_size_t("v4l2_pix_format.quantization", offsetof(struct v4l2_pix_format, quantization), &first);
    print_size_t("v4l2_pix_format.xfer_func", offsetof(struct v4l2_pix_format, xfer_func), &first);

    /* struct v4l2_format */
    print_size_t("v4l2_format.type", offsetof(struct v4l2_format, type), &first);
    print_size_t("v4l2_format.fmt", offsetof(struct v4l2_format, fmt), &first);
    print_size_t("v4l2_format.fmt.pix", offsetof(struct v4l2_format, fmt.pix), &first);

    /* struct v4l2_requestbuffers */
    print_size_t("v4l2_requestbuffers.count", offsetof(struct v4l2_requestbuffers, count), &first);
    print_size_t("v4l2_requestbuffers.type", offsetof(struct v4l2_requestbuffers, type), &first);
    print_size_t("v4l2_requestbuffers.memory", offsetof(struct v4l2_requestbuffers, memory), &first);
#ifdef V4L2_BUF_CAP_SUPPORTS_MMAP /* capabilities field is fairly recent */
    print_size_t("v4l2_requestbuffers.capabilities", offsetof(struct v4l2_requestbuffers, capabilities), &first);
#endif
#ifdef V4L2_MEMORY_FLAG_NON_COHERENT /* flags/reserved[3] added alongside the non-coherent flag */
    print_size_t("v4l2_requestbuffers.flags", offsetof(struct v4l2_requestbuffers, flags), &first);
    print_size_t("v4l2_requestbuffers.reserved", offsetof(struct v4l2_requestbuffers, reserved), &first);
#endif

    /* struct v4l2_buffer */
    print_size_t("v4l2_buffer.index", offsetof(struct v4l2_buffer, index), &first);
    print_size_t("v4l2_buffer.type", offsetof(struct v4l2_buffer, type), &first);
    print_size_t("v4l2_buffer.bytesused", offsetof(struct v4l2_buffer, bytesused), &first);
    print_size_t("v4l2_buffer.flags", offsetof(struct v4l2_buffer, flags), &first);
    print_size_t("v4l2_buffer.field", offsetof(struct v4l2_buffer, field), &first);
    print_size_t("v4l2_buffer.timestamp", offsetof(struct v4l2_buffer, timestamp), &first);
    print_size_t("v4l2_buffer.timecode", offsetof(struct v4l2_buffer, timecode), &first);
    print_size_t("v4l2_buffer.sequence", offsetof(struct v4l2_buffer, sequence), &first);
    print_size_t("v4l2_buffer.memory", offsetof(struct v4l2_buffer, memory), &first);
    print_size_t("v4l2_buffer.m", offsetof(struct v4l2_buffer, m), &first);
    print_size_t("v4l2_buffer.m.offset", offsetof(struct v4l2_buffer, m.offset), &first);
    print_size_t("v4l2_buffer.m.userptr", offsetof(struct v4l2_buffer, m.userptr), &first);
    print_size_t("v4l2_buffer.m.planes", offsetof(struct v4l2_buffer, m.planes), &first);
    print_size_t("v4l2_buffer.m.fd", offsetof(struct v4l2_buffer, m.fd), &first);
    print_size_t("v4l2_buffer.length", offsetof(struct v4l2_buffer, length), &first);
    print_size_t("v4l2_buffer.reserved2", offsetof(struct v4l2_buffer, reserved2), &first);
#ifdef V4L2_BUF_FLAG_REQUEST_FD /* request_fd/reserved anonymous union came with the Request API */
    print_size_t("v4l2_buffer.request_fd", offsetof(struct v4l2_buffer, request_fd), &first);
    print_size_t("v4l2_buffer.reserved", offsetof(struct v4l2_buffer, reserved), &first);
#endif

    /* struct v4l2_queryctrl */
    print_size_t("v4l2_queryctrl.id", offsetof(struct v4l2_queryctrl, id), &first);
    print_size_t("v4l2_queryctrl.type", offsetof(struct v4l2_queryctrl, type), &first);
    print_size_t("v4l2_queryctrl.name", offsetof(struct v4l2_queryctrl, name), &first);
    print_size_t("v4l2_queryctrl.minimum", offsetof(struct v4l2_queryctrl, minimum), &first);
    print_size_t("v4l2_queryctrl.maximum", offsetof(struct v4l2_queryctrl, maximum), &first);
    print_size_t("v4l2_queryctrl.step", offsetof(struct v4l2_queryctrl, step), &first);
    print_size_t("v4l2_queryctrl.default_value", offsetof(struct v4l2_queryctrl, default_value), &first);
    print_size_t("v4l2_queryctrl.flags", offsetof(struct v4l2_queryctrl, flags), &first);
    print_size_t("v4l2_queryctrl.reserved", offsetof(struct v4l2_queryctrl, reserved), &first);

    /* struct v4l2_control */
    print_size_t("v4l2_control.id", offsetof(struct v4l2_control, id), &first);
    print_size_t("v4l2_control.value", offsetof(struct v4l2_control, value), &first);

    /* struct v4l2_timecode */
    print_size_t("v4l2_timecode.type", offsetof(struct v4l2_timecode, type), &first);
    print_size_t("v4l2_timecode.flags", offsetof(struct v4l2_timecode, flags), &first);
    print_size_t("v4l2_timecode.frames", offsetof(struct v4l2_timecode, frames), &first);
    print_size_t("v4l2_timecode.seconds", offsetof(struct v4l2_timecode, seconds), &first);
    print_size_t("v4l2_timecode.minutes", offsetof(struct v4l2_timecode, minutes), &first);
    print_size_t("v4l2_timecode.hours", offsetof(struct v4l2_timecode, hours), &first);
    print_size_t("v4l2_timecode.userbits", offsetof(struct v4l2_timecode, userbits), &first);

    printf("\n  },\n");
}

/* ---------------------------------------------------------------------
 * ioctl request numbers. These are compile-time constants built by the
 * _IOR/_IOW/_IOWR macros from asm-generic/ioctl.h, so both decimal and
 * hex forms are printed to make it trivial to compare against whatever
 * base Python's own encoder produces.
 */
static void print_ioctl_value(const char *name, unsigned long value, int *first) {
    comma(first);
    printf("    \"%s\": {\"dec\": %lu, \"hex\": \"0x%08lX\"}", name, value, value);
}

static void print_ioctls(void) {
    int first = 1;
    printf("  \"ioctl\": {\n");
    print_ioctl_value("VIDIOC_QUERYCAP", (unsigned long)VIDIOC_QUERYCAP, &first);
    print_ioctl_value("VIDIOC_G_FMT", (unsigned long)VIDIOC_G_FMT, &first);
    print_ioctl_value("VIDIOC_S_FMT", (unsigned long)VIDIOC_S_FMT, &first);
    print_ioctl_value("VIDIOC_REQBUFS", (unsigned long)VIDIOC_REQBUFS, &first);
    print_ioctl_value("VIDIOC_QUERYBUF", (unsigned long)VIDIOC_QUERYBUF, &first);
    print_ioctl_value("VIDIOC_QBUF", (unsigned long)VIDIOC_QBUF, &first);
    print_ioctl_value("VIDIOC_DQBUF", (unsigned long)VIDIOC_DQBUF, &first);
    print_ioctl_value("VIDIOC_STREAMON", (unsigned long)VIDIOC_STREAMON, &first);
    print_ioctl_value("VIDIOC_STREAMOFF", (unsigned long)VIDIOC_STREAMOFF, &first);
    print_ioctl_value("VIDIOC_QUERYCTRL", (unsigned long)VIDIOC_QUERYCTRL, &first);
    print_ioctl_value("VIDIOC_G_CTRL", (unsigned long)VIDIOC_G_CTRL, &first);
    printf("\n  },\n");
}

/* ---------------------------------------------------------------------
 * Plain enum/#define constants the Python code compares buffer flags
 * and pixel formats against.
 */
static void print_constant(const char *name, unsigned long value, int *first) {
    comma(first);
    printf("    \"%s\": {\"dec\": %lu, \"hex\": \"0x%08lX\"}", name, value, value);
}

static void print_constants(void) {
    int first = 1;
    printf("  \"constants\": {\n");
    print_constant("V4L2_BUF_TYPE_VIDEO_CAPTURE", (unsigned long)V4L2_BUF_TYPE_VIDEO_CAPTURE, &first);
    print_constant("V4L2_MEMORY_MMAP", (unsigned long)V4L2_MEMORY_MMAP, &first);
    print_constant("V4L2_CAP_VIDEO_CAPTURE", (unsigned long)V4L2_CAP_VIDEO_CAPTURE, &first);
    print_constant("V4L2_CAP_STREAMING", (unsigned long)V4L2_CAP_STREAMING, &first);
#ifdef V4L2_CAP_DEVICE_CAPS
    print_constant("V4L2_CAP_DEVICE_CAPS", (unsigned long)V4L2_CAP_DEVICE_CAPS, &first);
#endif
#ifdef V4L2_CAP_META_CAPTURE
    print_constant("V4L2_CAP_META_CAPTURE", (unsigned long)V4L2_CAP_META_CAPTURE, &first);
#endif
    print_constant("V4L2_FIELD_NONE", (unsigned long)V4L2_FIELD_NONE, &first);
    print_constant("V4L2_PIX_FMT_Y16", (unsigned long)V4L2_PIX_FMT_Y16, &first);
    print_constant("V4L2_BUF_FLAG_MAPPED", (unsigned long)V4L2_BUF_FLAG_MAPPED, &first);
    print_constant("V4L2_BUF_FLAG_QUEUED", (unsigned long)V4L2_BUF_FLAG_QUEUED, &first);
    print_constant("V4L2_BUF_FLAG_DONE", (unsigned long)V4L2_BUF_FLAG_DONE, &first);
    print_constant("V4L2_BUF_FLAG_ERROR", (unsigned long)V4L2_BUF_FLAG_ERROR, &first);
#ifdef V4L2_BUF_FLAG_TIMESTAMP_MASK
    print_constant("V4L2_BUF_FLAG_TIMESTAMP_MASK", (unsigned long)V4L2_BUF_FLAG_TIMESTAMP_MASK, &first);
#endif
#ifdef V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
    print_constant("V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC", (unsigned long)V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, &first);
#endif
#ifdef V4L2_BUF_FLAG_TIMESTAMP_COPY
    print_constant("V4L2_BUF_FLAG_TIMESTAMP_COPY", (unsigned long)V4L2_BUF_FLAG_TIMESTAMP_COPY, &first);
#endif
#ifdef V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN
    print_constant("V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN", (unsigned long)V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN, &first);
#endif
/* The timestamp *source* field is a separate mask from the timestamp *type*
 * mask above; TSTAMP_SRC_EOF is zero, so only the mask makes it readable. */
#ifdef V4L2_BUF_FLAG_TSTAMP_SRC_MASK
    print_constant("V4L2_BUF_FLAG_TSTAMP_SRC_MASK", (unsigned long)V4L2_BUF_FLAG_TSTAMP_SRC_MASK, &first);
#endif
#ifdef V4L2_BUF_FLAG_TSTAMP_SRC_EOF
    print_constant("V4L2_BUF_FLAG_TSTAMP_SRC_EOF", (unsigned long)V4L2_BUF_FLAG_TSTAMP_SRC_EOF, &first);
#endif
#ifdef V4L2_BUF_FLAG_TSTAMP_SRC_SOE
    print_constant("V4L2_BUF_FLAG_TSTAMP_SRC_SOE", (unsigned long)V4L2_BUF_FLAG_TSTAMP_SRC_SOE, &first);
#endif
#ifdef V4L2_BUF_FLAG_LAST
    print_constant("V4L2_BUF_FLAG_LAST", (unsigned long)V4L2_BUF_FLAG_LAST, &first);
#endif
    print_constant("V4L2_CID_GAIN", (unsigned long)V4L2_CID_GAIN, &first);
    print_constant("V4L2_CID_CAMERA_CLASS_BASE", (unsigned long)V4L2_CID_CAMERA_CLASS_BASE, &first);
    print_constant("V4L2_CID_EXPOSURE_AUTO", (unsigned long)V4L2_CID_EXPOSURE_AUTO, &first);
    print_constant("V4L2_CID_EXPOSURE_ABSOLUTE", (unsigned long)V4L2_CID_EXPOSURE_ABSOLUTE, &first);
    print_constant("V4L2_CID_EXPOSURE_AUTO_PRIORITY", (unsigned long)V4L2_CID_EXPOSURE_AUTO_PRIORITY, &first);
    print_constant("V4L2_CID_AUTO_WHITE_BALANCE", (unsigned long)V4L2_CID_AUTO_WHITE_BALANCE, &first);
    print_constant("V4L2_CID_WHITE_BALANCE_TEMPERATURE", (unsigned long)V4L2_CID_WHITE_BALANCE_TEMPERATURE, &first);
    print_constant("V4L2_CTRL_TYPE_INTEGER", (unsigned long)V4L2_CTRL_TYPE_INTEGER, &first);
    print_constant("V4L2_CTRL_TYPE_BOOLEAN", (unsigned long)V4L2_CTRL_TYPE_BOOLEAN, &first);
    print_constant("V4L2_CTRL_TYPE_MENU", (unsigned long)V4L2_CTRL_TYPE_MENU, &first);
    print_constant("V4L2_CTRL_TYPE_INTEGER_MENU", (unsigned long)V4L2_CTRL_TYPE_INTEGER_MENU, &first);
    print_constant("V4L2_CTRL_FLAG_DISABLED", (unsigned long)V4L2_CTRL_FLAG_DISABLED, &first);
    print_constant("V4L2_CTRL_FLAG_GRABBED", (unsigned long)V4L2_CTRL_FLAG_GRABBED, &first);
    print_constant("V4L2_CTRL_FLAG_READ_ONLY", (unsigned long)V4L2_CTRL_FLAG_READ_ONLY, &first);
    print_constant("V4L2_CTRL_FLAG_UPDATE", (unsigned long)V4L2_CTRL_FLAG_UPDATE, &first);
    print_constant("V4L2_CTRL_FLAG_INACTIVE", (unsigned long)V4L2_CTRL_FLAG_INACTIVE, &first);
    print_constant("V4L2_CTRL_FLAG_SLIDER", (unsigned long)V4L2_CTRL_FLAG_SLIDER, &first);
    print_constant("V4L2_CTRL_FLAG_WRITE_ONLY", (unsigned long)V4L2_CTRL_FLAG_WRITE_ONLY, &first);
    print_constant("V4L2_CTRL_FLAG_VOLATILE", (unsigned long)V4L2_CTRL_FLAG_VOLATILE, &first);
    printf("\n  },\n");
}

/* ---------------------------------------------------------------------
 * V4L2_PIX_FMT_Y16 is built via the v4l2_fourcc() macro from 4 chars;
 * decode it back to a string here to sanity-check it really is "Y16 "
 * (note the trailing space) rather than trusting the macro algebra by
 * eye when reading Python code that reproduces the same fourcc.
 */
static void print_fourcc_check(void) {
    __u32 fourcc = V4L2_PIX_FMT_Y16;
    char text[5];
    text[0] = (char)(fourcc & 0xFF);
    text[1] = (char)((fourcc >> 8) & 0xFF);
    text[2] = (char)((fourcc >> 16) & 0xFF);
    text[3] = (char)((fourcc >> 24) & 0xFF);
    text[4] = '\0';
    /* Escape nothing special is expected here (ASCII letters/space/digits). */
    printf("  \"V4L2_PIX_FMT_Y16_fourcc_text\": \"%s\",\n", text);
}

/* ---------------------------------------------------------------------
 * _IOC encoding parameters (step 5): report these so the Python ioctl
 * request-number builder can be checked bit-for-bit against the macros
 * actually used to build VIDIOC_* above.
 */
static void print_ioc_encoding(void) {
    printf("  \"ioc_encoding\": {\n");
    printf("    \"_IOC_NRBITS\": %d,\n", _IOC_NRBITS);
    printf("    \"_IOC_TYPEBITS\": %d,\n", _IOC_TYPEBITS);
    printf("    \"_IOC_SIZEBITS\": %d,\n", _IOC_SIZEBITS);
    printf("    \"_IOC_DIRBITS\": %d,\n", _IOC_DIRBITS);
    printf("    \"_IOC_NRSHIFT\": %d,\n", _IOC_NRSHIFT);
    printf("    \"_IOC_TYPESHIFT\": %d,\n", _IOC_TYPESHIFT);
    printf("    \"_IOC_SIZESHIFT\": %d,\n", _IOC_SIZESHIFT);
    printf("    \"_IOC_DIRSHIFT\": %d,\n", _IOC_DIRSHIFT);
    printf("    \"_IOC_NONE\": %u,\n", (unsigned)_IOC_NONE);
    printf("    \"_IOC_WRITE\": %u,\n", (unsigned)_IOC_WRITE);
    printf("    \"_IOC_READ\": %u\n", (unsigned)_IOC_READ);
    printf("  }\n");
}

int main(void) {
    printf("{\n");
    print_sizeofs();
    print_offsets();
    print_ioctls();
    print_constants();
    print_fourcc_check();
    print_ioc_encoding();
    printf("}\n");
    return 0;
}
