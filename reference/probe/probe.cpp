// probe.cpp - standalone reference probe against the Revopoint 3DCamera SDK.
//
// Purpose: exercise the read-only / query surface of the SDK (namespace cs)
// against a real, attached camera and emit a single JSON object describing
// everything the SDK told us, plus a small, time-bounded frame capture to
// prove streaming works end to end. This file is deliberately independent
// of the Python package in this repository; it only *references* the SDK
// by path (see build.sh) and is not linked into anything else.
//
// Hard safety properties (see also README.md):
//   - Never runs longer than --timeout-ms (default 12000) wall clock: a
//     watchdog thread force-exits the process with a partial JSON dump if
//     the main thread has not finished by then.
//   - Every SDK call's ERROR_CODE is checked and recorded.
//   - stopStream()/disconnect() are attempted on every normal (non-forced)
//     exit path via RAII scope guards.
//   - Only queries + start/stop stream are used. No firmware upgrade, no
//     persistent setting changes.

#include "3DCamera.hpp"

// Legacy C API (createSystem/systemConnectCamera/...). Not pulled in by
// 3DCamera.hpp itself; only used by the --c-api-connect diagnostic path
// added to investigate the ERROR_NOT_SUPPORT finding. CS_API is already
// defined by the time these are included (via 3DCamera.hpp above).
#include "h/system.h"
#include "h/camera.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#ifndef SDK_LIB_PATH_STR
#define SDK_LIB_PATH_STR "unknown"
#endif

using namespace cs;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string &in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

static std::string jstr(const std::string &s) {
    return "\"" + json_escape(s) + "\"";
}

static std::string cstr_field(const char *p, size_t maxlen) {
    // CameraInfo fields are fixed-size char arrays that may not be
    // NUL-terminated if the SDK ever filled them exactly; be defensive.
    size_t n = 0;
    while (n < maxlen && p[n] != '\0') n++;
    return std::string(p, n);
}

static std::string error_code_name(ERROR_CODE code) {
    switch (code) {
        case SUCCESS: return "SUCCESS";
        case ERROR_PARAM: return "ERROR_PARAM";
        case ERROR_DEVICE_NOT_FOUND: return "ERROR_DEVICE_NOT_FOUND";
        case ERROR_DEVICE_NOT_CONNECT: return "ERROR_DEVICE_NOT_CONNECT";
        case ERROR_DEVICE_BUSY: return "ERROR_DEVICE_BUSY";
        case ERROR_STREAM_NOT_START: return "ERROR_STREAM_NOT_START";
        case ERROR_STREAM_BUSY: return "ERROR_STREAM_BUSY";
        case ERROR_FRAME_TIMEOUT: return "ERROR_FRAME_TIMEOUT";
        case ERROR_NOT_SUPPORT: return "ERROR_NOT_SUPPORT";
        case ERROR_PROPERTY_GET_FAILED: return "ERROR_PROPERTY_GET_FAILED";
        case ERROR_PROPERTY_SET_FAILED: return "ERROR_PROPERTY_SET_FAILED";
        case ERROR_HID_CHANNEL_ERROR: return "ERROR_HID_CHANNEL_ERROR";
        case ERROR_HID_WRITE_ERROR: return "ERROR_HID_WRITE_ERROR";
        case ERROR_HID_READ_ERROR: return "ERROR_HID_READ_ERROR";
        case ERROR_UNKNOW: return "ERROR_UNKNOW";
        default: return "ERROR_CODE(" + std::to_string((int)code) + ")";
    }
}

static std::string stream_format_name(int fmt) {
    switch (fmt) {
        case STREAM_FORMAT_MJPG: return "MJPG";
        case STREAM_FORMAT_RGB8: return "RGB8";
        case STREAM_FORMAT_Z16: return "Z16";
        case STREAM_FORMAT_Z16Y8Y8: return "Z16Y8Y8";
        case STREAM_FORMAT_PAIR: return "PAIR";
        case STREAM_FORMAT_H264: return "H264";
        case STREAM_FORMAT_I8DS: return "I8DS";
        case STREAM_FORMAT_XZ32: return "XZ32";
        case STREAM_FORMAT_GRAY: return "GRAY";
        default: return "STREAM_FORMAT(0x" + [&]{
            std::ostringstream o; o << std::hex << fmt; return o.str();
        }() + ")";
    }
}

// ---------------------------------------------------------------------------
// Probe state - single source of truth for the JSON document. Guarded by a
// mutex so both the main thread and the watchdog thread can safely read it
// (the watchdog only ever reads/serializes; the main thread is the sole
// writer).
// ---------------------------------------------------------------------------

struct StreamProfileRec {
    int format = 0;
    std::string format_name;
    int width = 0;
    int height = 0;
    float fps = 0;
};

struct IntrinsicsRec {
    bool ok = false;
    short width = 0, height = 0;
    float fx = 0, zero01 = 0, cx = 0, zeor10 = 0, fy = 0, cy = 0, zeor20 = 0, zero21 = 0, one22 = 0;
};

struct DistortRec {
    bool ok = false;
    float k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0;
};

struct FrameStats {
    int count = 0;
    int width = 0, height = 0;
    int format = -1;
    std::string format_name;
    int size_bytes = 0;
    bool have_ts = false;
    double first_ts = 0, last_ts = 0;
};

struct ErrEntry {
    std::string stage;
    int code = 0;
    std::string code_name;
    std::string message;
};

struct ProbeState {
    std::mutex mu;

    std::string sdk_version = "unknown";
    std::string sdk_lib_path = SDK_LIB_PATH_STR;
    std::string requested_serial;

    struct CamRec {
        std::string name, serial, uniqueId, firmwareVersion, algorithmVersion;
    };
    std::vector<CamRec> cameras_found;

    bool matched = false;
    std::string model;
    std::string camera_type;

    bool connect_attempted = false;
    bool connect_ok = false;

    bool stream_support_depth_ok = false, stream_support_depth = false;
    bool stream_support_rgb_ok = false, stream_support_rgb = false;

    bool depth_profiles_ok = false, rgb_profiles_ok = false;
    std::vector<StreamProfileRec> depth_profiles, rgb_profiles;
    bool chosen_depth_set = false, chosen_rgb_set = false;
    StreamProfileRec chosen_depth, chosen_rgb;

    IntrinsicsRec intrinsics_depth, intrinsics_rgb;
    bool extrinsics_ok = false;
    float extrinsics_rotation[9] = {0};
    float extrinsics_translation[3] = {0};
    DistortRec distort_depth, distort_rgb;

    bool depth_scale_ok = false;
    float depth_scale = 0;

    bool depth_stream_started = false, rgb_stream_started = false;
    FrameStats depth_frames, rgb_frames;

    std::vector<ErrEntry> errors;

    std::string status = "running";
    long elapsed_ms = 0;

    void add_error(const std::string &stage, ERROR_CODE code, const std::string &message) {
        std::lock_guard<std::mutex> lk(mu);
        ErrEntry e;
        e.stage = stage;
        e.code = (int)code;
        e.code_name = error_code_name(code);
        e.message = message;
        errors.push_back(e);
    }
};

static ProbeState g_state;
static Clock::time_point g_start;
static std::atomic<bool> g_done{false};
static long g_timeout_ms = 12000;
static std::string g_json_out_path;

// --- Diagnostic-only additions (all default to "off", preserving the
// original default behaviour byte-for-byte when none of these flags are
// passed). Added to investigate the ERROR_NOT_SUPPORT finding on connect().
static bool g_sdk_log_set = false;
static std::string g_sdk_log_dir;
static bool g_libuvc_set = false;
static bool g_libuvc_value = false;
static bool g_checked_inited_set = false;
static bool g_checked_inited_value = false;
static bool g_init_driver_set = false;
static std::string g_init_driver_code;
static std::string g_connect_mode = "info"; // "info" (default) or "noargs"
static bool g_c_api_mode = false;

static long elapsed_ms_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_start).count();
}

// Records an SDK call's result; returns true if SUCCESS.
static bool check(const std::string &stage, ERROR_CODE code) {
    if (code != SUCCESS) {
        g_state.add_error(stage, code, std::string(getCameraErrorString(code) ? getCameraErrorString(code) : ""));
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// JSON serialization (reads g_state under lock; safe to call from either
// thread at any point in the run, producing whatever has been filled in
// so far).
// ---------------------------------------------------------------------------

static std::string build_json_locked() {
    std::ostringstream o;
    o << "{\n";
    o << "  \"sdk_version\": " << jstr(g_state.sdk_version) << ",\n";
    o << "  \"sdk_lib_path\": " << jstr(g_state.sdk_lib_path) << ",\n";
    o << "  \"requested_serial\": " << jstr(g_state.requested_serial) << ",\n";

    o << "  \"cameras_found\": [";
    for (size_t i = 0; i < g_state.cameras_found.size(); i++) {
        const auto &c = g_state.cameras_found[i];
        if (i) o << ",";
        o << "\n    {\"name\": " << jstr(c.name)
          << ", \"serial\": " << jstr(c.serial)
          << ", \"uniqueId\": " << jstr(c.uniqueId)
          << ", \"firmwareVersion\": " << jstr(c.firmwareVersion)
          << ", \"algorithmVersion\": " << jstr(c.algorithmVersion) << "}";
    }
    o << (g_state.cameras_found.empty() ? "" : "\n  ") << "],\n";

    o << "  \"matched\": " << (g_state.matched ? "true" : "false") << ",\n";
    o << "  \"model\": " << (g_state.model.empty() ? "null" : jstr(g_state.model)) << ",\n";
    o << "  \"camera_type\": " << (g_state.camera_type.empty() ? "null" : jstr(g_state.camera_type)) << ",\n";

    o << "  \"connect\": " << (g_state.connect_attempted ? (g_state.connect_ok ? "true" : "false") : "null") << ",\n";

    o << "  \"stream_support\": {\"depth\": "
      << (g_state.stream_support_depth_ok ? (g_state.stream_support_depth ? "true" : "false") : "null")
      << ", \"rgb\": "
      << (g_state.stream_support_rgb_ok ? (g_state.stream_support_rgb ? "true" : "false") : "null")
      << "},\n";

    auto emit_profiles = [&](const std::vector<StreamProfileRec> &v) {
        o << "[";
        for (size_t i = 0; i < v.size(); i++) {
            const auto &p = v[i];
            if (i) o << ",";
            o << "\n      {\"format\": " << p.format
              << ", \"format_name\": " << jstr(p.format_name)
              << ", \"width\": " << p.width
              << ", \"height\": " << p.height
              << ", \"fps\": " << p.fps << "}";
        }
        o << (v.empty() ? "" : "\n    ") << "]";
    };
    o << "  \"stream_profiles\": {\n";
    o << "    \"depth\": "; emit_profiles(g_state.depth_profiles); o << ",\n";
    o << "    \"rgb\": "; emit_profiles(g_state.rgb_profiles); o << "\n";
    o << "  },\n";

    auto emit_chosen = [&](bool set, const StreamProfileRec &p) {
        if (!set) { o << "null"; return; }
        o << "{\"format\": " << p.format << ", \"format_name\": " << jstr(p.format_name)
          << ", \"width\": " << p.width << ", \"height\": " << p.height << ", \"fps\": " << p.fps << "}";
    };
    o << "  \"chosen_stream_profiles\": {\"depth\": ";
    emit_chosen(g_state.chosen_depth_set, g_state.chosen_depth);
    o << ", \"rgb\": ";
    emit_chosen(g_state.chosen_rgb_set, g_state.chosen_rgb);
    o << "},\n";

    auto emit_intrinsics = [&](const IntrinsicsRec &r) {
        if (!r.ok) { o << "null"; return; }
        o << "{\"width\": " << r.width << ", \"height\": " << r.height
          << ", \"fx\": " << r.fx << ", \"zero01\": " << r.zero01 << ", \"cx\": " << r.cx
          << ", \"zeor10\": " << r.zeor10 << ", \"fy\": " << r.fy << ", \"cy\": " << r.cy
          << ", \"zeor20\": " << r.zeor20 << ", \"zero21\": " << r.zero21 << ", \"one22\": " << r.one22 << "}";
    };
    o << "  \"intrinsics\": {\"depth\": ";
    emit_intrinsics(g_state.intrinsics_depth);
    o << ", \"rgb\": ";
    emit_intrinsics(g_state.intrinsics_rgb);
    o << "},\n";

    o << "  \"extrinsics\": ";
    if (g_state.extrinsics_ok) {
        o << "{\"rotation\": [";
        for (int i = 0; i < 9; i++) { if (i) o << ", "; o << g_state.extrinsics_rotation[i]; }
        o << "], \"translation\": [";
        for (int i = 0; i < 3; i++) { if (i) o << ", "; o << g_state.extrinsics_translation[i]; }
        o << "]}";
    } else {
        o << "null";
    }
    o << ",\n";

    auto emit_distort = [&](const DistortRec &r) {
        if (!r.ok) { o << "null"; return; }
        o << "{\"k1\": " << r.k1 << ", \"k2\": " << r.k2 << ", \"k3\": " << r.k3
          << ", \"k4\": " << r.k4 << ", \"k5\": " << r.k5 << "}";
    };
    o << "  \"distort\": {\"depth\": ";
    emit_distort(g_state.distort_depth);
    o << ", \"rgb\": ";
    emit_distort(g_state.distort_rgb);
    o << "},\n";

    o << "  \"depth_scale\": " << (g_state.depth_scale_ok ? std::to_string(g_state.depth_scale) : "null") << ",\n";

    auto emit_frame_stats = [&](const FrameStats &f) {
        if (f.count == 0) {
            o << "{\"count\": 0, \"width\": null, \"height\": null, \"format\": null, \"format_name\": null, "
                 "\"size_bytes\": null, \"first_timestamp_ms\": null, \"last_timestamp_ms\": null, \"observed_fps\": null}";
            return;
        }
        double observed_fps = 0.0;
        if (f.have_ts && f.count > 1 && f.last_ts > f.first_ts) {
            observed_fps = (double)(f.count - 1) * 1000.0 / (f.last_ts - f.first_ts);
        }
        o << "{\"count\": " << f.count
          << ", \"width\": " << f.width
          << ", \"height\": " << f.height
          << ", \"format\": " << f.format
          << ", \"format_name\": " << jstr(f.format_name)
          << ", \"size_bytes\": " << f.size_bytes
          << ", \"first_timestamp_ms\": " << f.first_ts
          << ", \"last_timestamp_ms\": " << f.last_ts
          << ", \"observed_fps\": " << observed_fps << "}";
    };
    o << "  \"frames\": {\"depth\": ";
    emit_frame_stats(g_state.depth_frames);
    o << ", \"rgb\": ";
    emit_frame_stats(g_state.rgb_frames);
    o << "},\n";

    o << "  \"errors\": [";
    for (size_t i = 0; i < g_state.errors.size(); i++) {
        const auto &e = g_state.errors[i];
        if (i) o << ",";
        o << "\n    {\"stage\": " << jstr(e.stage)
          << ", \"code\": " << e.code
          << ", \"code_name\": " << jstr(e.code_name)
          << ", \"message\": " << jstr(e.message) << "}";
    }
    o << (g_state.errors.empty() ? "" : "\n  ") << "],\n";

    o << "  \"elapsed_ms\": " << g_state.elapsed_ms << ",\n";
    o << "  \"status\": " << jstr(g_state.status) << "\n";
    o << "}\n";
    return o.str();
}

static void emit_json(const std::string &json) {
    std::fputs(json.c_str(), stdout);
    std::fflush(stdout);
    if (!g_json_out_path.empty()) {
        FILE *f = std::fopen(g_json_out_path.c_str(), "w");
        if (f) {
            std::fwrite(json.data(), 1, json.size(), f);
            std::fclose(f);
        } else {
            std::fprintf(stderr, "warning: could not open --json-out path %s for writing\n", g_json_out_path.c_str());
        }
    }
}

// Watchdog: fires only if the main thread has not finished within the hard
// budget. It cannot safely call back into the SDK (the main thread may be
// blocked inside a stuck SDK call), so on timeout it does *not* attempt
// stopStream/disconnect -- it only serializes the best-known state and
// force-exits. This is documented as the one exception to "cleanup on every
// exit path".
static void watchdog_thread_fn() {
    const auto deadline = g_start + std::chrono::milliseconds(g_timeout_ms);
    while (Clock::now() < deadline) {
        if (g_done.load()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (g_done.load()) return;
    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.status = "timeout";
        g_state.elapsed_ms = elapsed_ms_now();
    }
    std::string json;
    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        json = build_json_locked();
    }
    std::fprintf(stderr, "probe: hard timeout of %ld ms reached, force-exiting without SDK cleanup\n", g_timeout_ms);
    emit_json(json);
    std::_Exit(4);
}

// ---------------------------------------------------------------------------
// RAII cleanup guard: stops any streams that were started and disconnects
// the camera. Safe to call multiple times / with nothing started.
// ---------------------------------------------------------------------------

struct CameraCleanup {
    ICameraPtr camera;
    bool *depth_started = nullptr;
    bool *rgb_started = nullptr;
    bool *connected = nullptr;
    ~CameraCleanup() {
        if (!camera) return;
        if (depth_started && *depth_started) {
            ERROR_CODE rc = camera->stopStream(STREAM_TYPE_DEPTH);
            check("cleanup.stopStream(depth)", rc);
            *depth_started = false;
        }
        if (rgb_started && *rgb_started) {
            ERROR_CODE rc = camera->stopStream(STREAM_TYPE_RGB);
            check("cleanup.stopStream(rgb)", rc);
            *rgb_started = false;
        }
        if (connected && *connected) {
            ERROR_CODE rc = camera->disconnect();
            check("cleanup.disconnect", rc);
            *connected = false;
        }
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s --serial <SN> [--json-out <path>] [--timeout-ms <int>] [--frames <n>]\n"
        "       [--sdk-log-dir <path>] [--libuvc true|false] [--checked-camera-inited true|false]\n"
        "       [--init-driver-code <code>] [--connect-mode info|noargs] [--c-api-connect]\n"
        "(the second line of flags are diagnostic-only additions; omitting all of them\n"
        " reproduces the original default behaviour exactly)\n",
        argv0);
}

// Legacy C API diagnostic: bypasses the whole cs:: C++ flow above and
// exercises systemConnectCamera() directly, to check whether the raw C
// entry point behaves differently from cs::ICamera::connect(). Because this
// API reports success/failure only as a null/non-null CCamera*, not an
// ERROR_CODE, we cannot record a symbolic error code for the connect call
// itself -- that limitation is reported verbatim rather than papered over.
static int run_c_api_diagnostic(const std::string &serial, long budget_ms) {
    auto t0 = Clock::now();
    auto ms_since = [&]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    };

    std::fprintf(stderr, "diagnose(c-api): createSystem()\n");
    CSystem *sys = createSystem();
    if (!sys) {
        std::fprintf(stderr, "diagnose(c-api): createSystem() returned null\n");
        return 5;
    }

    // systemCreateCameraInfoList (unlike cs::ISystem::queryCameras) takes no
    // timeout and returns whatever the async USB-discovery thread has found
    // so far, which can be empty immediately after createSystem(). Poll a
    // few times with short sleeps, staying well inside the overall budget.
    int count = 0;
    CameraInfo *list = nullptr;
    for (int attempt = 0; attempt < 20; attempt++) {
        if (list) systemDeleteCameraInfoList(list);
        list = systemCreateCameraInfoList(sys, &count);
        if (count > 0) break;
        if (ms_since() > budget_ms - 2000) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    std::fprintf(stderr, "diagnose(c-api): systemCreateCameraInfoList -> count=%d (after %lldms)\n",
                  count, (long long)ms_since());

    int matchIdx = -1;
    for (int i = 0; i < count; i++) {
        std::string s = cstr_field(list[i].serial, sizeof(list[i].serial));
        std::fprintf(stderr, "diagnose(c-api): camera[%d] serial=%s name=%s\n", i, s.c_str(),
                      cstr_field(list[i].name, sizeof(list[i].name)).c_str());
        if (s == serial) matchIdx = i;
    }

    int rc_code = 2;
    if (matchIdx < 0) {
        std::fprintf(stderr, "diagnose(c-api): requested serial %s not found\n", serial.c_str());
    } else {
        CameraInfo info = list[matchIdx];
        std::fprintf(stderr, "diagnose(c-api): systemConnectCamera(...) elapsed_before=%lldms\n",
                      (long long)ms_since());
        CCamera *camera = systemConnectCamera(sys, &info);
        std::fprintf(stderr, "diagnose(c-api): systemConnectCamera -> %s (elapsed=%lldms)\n",
                      camera ? "non-null (connected)" : "NULL (failed)", (long long)ms_since());
        rc_code = camera ? 0 : 3;
        if (camera) {
            PropertyExtension pe{};
            ERROR_CODE r = cameraGetPropertyExtension(camera, PROPERTY_EXT_DEPTH_SCALE, &pe);
            std::fprintf(stderr, "diagnose(c-api): cameraGetPropertyExtension(DEPTH_SCALE) -> %s (%d) depthScale=%f\n",
                         error_code_name(r).c_str(), (int)r, pe.depthScale);
            systemDisconnectCamera(sys, camera);
            std::fprintf(stderr, "diagnose(c-api): systemDisconnectCamera done\n");
        }
    }

    if (list) systemDeleteCameraInfoList(list);
    deleteSystem(sys);
    std::fprintf(stderr, "diagnose(c-api): cleanup done, total elapsed=%lldms (budget=%ldms)\n",
                  (long long)ms_since(), budget_ms);
    return rc_code;
}

int main(int argc, char **argv) {
    g_start = Clock::now();

    std::string serial;
    long frames_requested = 5;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--serial") serial = next("--serial");
        else if (a == "--json-out") g_json_out_path = next("--json-out");
        else if (a == "--timeout-ms") g_timeout_ms = std::strtol(next("--timeout-ms").c_str(), nullptr, 10);
        else if (a == "--frames") frames_requested = std::strtol(next("--frames").c_str(), nullptr, 10);
        else if (a == "--sdk-log-dir") { g_sdk_log_set = true; g_sdk_log_dir = next("--sdk-log-dir"); }
        else if (a == "--libuvc") { g_libuvc_set = true; std::string v = next("--libuvc"); g_libuvc_value = (v == "true" || v == "1"); }
        else if (a == "--checked-camera-inited") { g_checked_inited_set = true; std::string v = next("--checked-camera-inited"); g_checked_inited_value = (v == "true" || v == "1"); }
        else if (a == "--init-driver-code") { g_init_driver_set = true; g_init_driver_code = next("--init-driver-code"); }
        else if (a == "--connect-mode") g_connect_mode = next("--connect-mode");
        else if (a == "--c-api-connect") g_c_api_mode = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (serial.empty()) {
        std::fprintf(stderr, "error: --serial is required\n");
        usage(argv[0]);
        return 1;
    }
    if (g_timeout_ms <= 0) g_timeout_ms = 12000;
    if (frames_requested <= 0) frames_requested = 5;

    // --- Diagnostic-only entry points -------------------------------------
    // These bypass the normal cs:: flow entirely and are only reached when
    // explicitly requested; the default invocation never touches this code.
    if (g_c_api_mode) {
        return run_c_api_diagnostic(serial, g_timeout_ms);
    }
    if (g_sdk_log_set) {
        ::mkdir(g_sdk_log_dir.c_str(), 0755); // best-effort; ignore EEXIST/errors
        setLogSavePath(g_sdk_log_dir.c_str());
        enableLoging(true);
        std::fprintf(stderr, "diagnose: SDK logging enabled, save path=%s\n", g_sdk_log_dir.c_str());
    }
    if (g_libuvc_set) {
        setSdkEnableLibuvc(g_libuvc_value);
        std::fprintf(stderr, "diagnose: setSdkEnableLibuvc(%s)\n", g_libuvc_value ? "true" : "false");
    }
    if (g_checked_inited_set) {
        setCheckedCameraInited(g_checked_inited_value);
        std::fprintf(stderr, "diagnose: setCheckedCameraInited(%s)\n", g_checked_inited_value ? "true" : "false");
    }

    g_state.requested_serial = serial;

    std::thread watchdog(watchdog_thread_fn);
    watchdog.detach();

    // Safety margin reserved for cleanup + JSON serialization before the
    // hard budget expires.
    const long kCleanupMarginMs = 500;
    auto remaining_ms = [&]() -> long {
        long r = g_timeout_ms - elapsed_ms_now() - kCleanupMarginMs;
        return r > 0 ? r : 0;
    };

    // --- SDK version -------------------------------------------------------
    {
        CS_SDK_VERSION *ver = nullptr;
        ERROR_CODE rc = getSdkVersion(&ver);
        if (check("getSdkVersion", rc) && ver) {
            std::lock_guard<std::mutex> lk(g_state.mu);
            std::string v = ver->version ? ver->version : "";
            std::string d = ver->date ? ver->date : "";
            g_state.sdk_version = d.empty() ? v : (v + " (" + d + ")");
        }
    }

    ISystemPtr system = getSystemPtr();
    if (!system) {
        g_state.add_error("getSystemPtr", ERROR_UNKNOW, "getSystemPtr() returned null");
        std::string json;
        {
            std::lock_guard<std::mutex> lk(g_state.mu);
            g_state.status = "sdk_error";
            g_state.elapsed_ms = elapsed_ms_now();
            json = build_json_locked();
        }
        emit_json(json);
        g_done = true;
        return 5;
    }

    if (g_init_driver_set) {
        ERROR_CODE r = system->initDriver(g_init_driver_code.c_str());
        check("initDriver(system)", r);
        std::fprintf(stderr, "diagnose: system->initDriver(\"%s\") -> %s (%d)\n",
                      g_init_driver_code.c_str(), error_code_name(r).c_str(), (int)r);
    }

    // --- queryCameras --------------------------------------------------------
    std::vector<CameraInfo> cameras;
    {
        long qtimeout = std::min<long>(3000, remaining_ms());
        ERROR_CODE rc = system->queryCameras(cameras, (int)qtimeout);
        check("queryCameras", rc);
        std::lock_guard<std::mutex> lk(g_state.mu);
        for (auto &c : cameras) {
            ProbeState::CamRec rec;
            rec.name = cstr_field(c.name, sizeof(c.name));
            rec.serial = cstr_field(c.serial, sizeof(c.serial));
            rec.uniqueId = cstr_field(c.uniqueId, sizeof(c.uniqueId));
            rec.firmwareVersion = cstr_field(c.firmwareVersion, sizeof(c.firmwareVersion));
            rec.algorithmVersion = cstr_field(c.algorithmVersion, sizeof(c.algorithmVersion));
            g_state.cameras_found.push_back(rec);
        }
    }

    CameraInfo matchedInfo{};
    bool matched = false;
    for (auto &c : cameras) {
        if (serial == cstr_field(c.serial, sizeof(c.serial))) {
            matchedInfo = c;
            matched = true;
            break;
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.matched = matched;
        if (matched) g_state.model = cstr_field(matchedInfo.name, sizeof(matchedInfo.name));
    }

    if (matched) {
        CameraType ct = getCameraTypeBySN(serial.c_str());
        const char *ctname = getCameraTypeName(ct);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.camera_type = ctname ? ctname : "";
    }

    auto finish = [&](const std::string &status, int code) -> int {
        std::string json;
        {
            std::lock_guard<std::mutex> lk(g_state.mu);
            g_state.status = status;
            g_state.elapsed_ms = elapsed_ms_now();
            json = build_json_locked();
        }
        emit_json(json);
        g_done = true;
        return code;
    };

    if (!matched) {
        std::fprintf(stderr, "probe: no camera with serial %s found among %zu enumerated camera(s)\n",
                      serial.c_str(), cameras.size());
        return finish("device_not_found", 2);
    }

    // --- connect -------------------------------------------------------------
    ICameraPtr camera = getCameraPtr();
    bool connected = false;
    bool depth_started = false, rgb_started = false;
    CameraCleanup cleanup;
    cleanup.camera = camera;
    cleanup.depth_started = &depth_started;
    cleanup.rgb_started = &rgb_started;
    cleanup.connected = &connected;

    if (!camera) {
        g_state.add_error("getCameraPtr", ERROR_UNKNOW, "getCameraPtr() returned null");
        return finish("sdk_error", 5);
    }

    if (g_init_driver_set) {
        ERROR_CODE r = camera->initDriver(g_init_driver_code.c_str());
        check("initDriver(camera)", r);
        std::fprintf(stderr, "diagnose: camera->initDriver(\"%s\") -> %s (%d)\n",
                      g_init_driver_code.c_str(), error_code_name(r).c_str(), (int)r);
    }

    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.connect_attempted = true;
    }
    ERROR_CODE rc = (g_connect_mode == "noargs") ? camera->connect() : camera->connect(matchedInfo);
    if (g_connect_mode != "info") {
        std::fprintf(stderr, "diagnose: connect-mode=%s -> %s (%d)\n", g_connect_mode.c_str(),
                     error_code_name(rc).c_str(), (int)rc);
    }
    bool connect_ok = check("connect", rc);
    connected = connect_ok;
    {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.connect_ok = connect_ok;
    }
    if (!connect_ok) {
        return finish("connect_failed", 3);
    }

    // --- getInfo (sanity) ------------------------------------------------
    {
        CameraInfo info{};
        ERROR_CODE r = camera->getInfo(info);
        check("getInfo", r);
    }

    // --- isStreamSupport ---------------------------------------------------
    {
        bool support = false;
        ERROR_CODE r = camera->isStreamSupport(STREAM_TYPE_DEPTH, support);
        bool ok = check("isStreamSupport(depth)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.stream_support_depth_ok = ok;
        g_state.stream_support_depth = support;
    }
    {
        bool support = false;
        ERROR_CODE r = camera->isStreamSupport(STREAM_TYPE_RGB, support);
        bool ok = check("isStreamSupport(rgb)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.stream_support_rgb_ok = ok;
        g_state.stream_support_rgb = support;
    }

    // --- getStreamInfos + profile selection --------------------------------
    std::vector<StreamInfo> depthInfos, rgbInfos;
    {
        ERROR_CODE r = camera->getStreamInfos(STREAM_TYPE_DEPTH, depthInfos);
        bool ok = check("getStreamInfos(depth)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.depth_profiles_ok = ok;
        for (auto &s : depthInfos) {
            StreamProfileRec p;
            p.format = (int)s.format;
            p.format_name = stream_format_name((int)s.format);
            p.width = s.width;
            p.height = s.height;
            p.fps = s.fps;
            g_state.depth_profiles.push_back(p);
        }
    }
    {
        ERROR_CODE r = camera->getStreamInfos(STREAM_TYPE_RGB, rgbInfos);
        bool ok = check("getStreamInfos(rgb)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.rgb_profiles_ok = ok;
        for (auto &s : rgbInfos) {
            StreamProfileRec p;
            p.format = (int)s.format;
            p.format_name = stream_format_name((int)s.format);
            p.width = s.width;
            p.height = s.height;
            p.fps = s.fps;
            g_state.rgb_profiles.push_back(p);
        }
    }

    auto pick_profile = [](const std::vector<StreamInfo> &infos, std::initializer_list<STREAM_FORMAT> preferred,
                            StreamInfo &out) -> bool {
        for (auto want : preferred) {
            for (auto &s : infos) {
                if (s.format == want) { out = s; return true; }
            }
        }
        if (!infos.empty()) { out = infos.front(); return true; }
        return false;
    };

    StreamInfo depthChoice{}, rgbChoice{};
    bool haveDepthChoice = pick_profile(depthInfos, {STREAM_FORMAT_Z16, STREAM_FORMAT_Z16Y8Y8}, depthChoice);
    bool haveRgbChoice = pick_profile(rgbInfos, {STREAM_FORMAT_MJPG, STREAM_FORMAT_RGB8}, rgbChoice);

    if (!haveDepthChoice) {
        g_state.add_error("profile_selection(depth)", ERROR_PARAM, "no depth stream profiles available");
    } else {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.chosen_depth_set = true;
        g_state.chosen_depth.format = (int)depthChoice.format;
        g_state.chosen_depth.format_name = stream_format_name((int)depthChoice.format);
        g_state.chosen_depth.width = depthChoice.width;
        g_state.chosen_depth.height = depthChoice.height;
        g_state.chosen_depth.fps = depthChoice.fps;
    }
    if (!haveRgbChoice) {
        g_state.add_error("profile_selection(rgb)", ERROR_PARAM, "no rgb stream profiles available");
    } else {
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.chosen_rgb_set = true;
        g_state.chosen_rgb.format = (int)rgbChoice.format;
        g_state.chosen_rgb.format_name = stream_format_name((int)rgbChoice.format);
        g_state.chosen_rgb.width = rgbChoice.width;
        g_state.chosen_rgb.height = rgbChoice.height;
        g_state.chosen_rgb.fps = rgbChoice.fps;
    }
    if (haveDepthChoice) {
        std::fprintf(stderr, "probe: chosen depth profile: format=%s width=%d height=%d fps=%.2f\n",
                      stream_format_name((int)depthChoice.format).c_str(), depthChoice.width, depthChoice.height, depthChoice.fps);
    } else {
        std::fprintf(stderr, "probe: no depth profile available to choose\n");
    }
    if (haveRgbChoice) {
        std::fprintf(stderr, "probe: chosen rgb profile: format=%s width=%d height=%d fps=%.2f\n",
                      stream_format_name((int)rgbChoice.format).c_str(), rgbChoice.width, rgbChoice.height, rgbChoice.fps);
    } else {
        std::fprintf(stderr, "probe: no rgb profile available to choose\n");
    }

    // --- intrinsics / extrinsics / distort / depth scale --------------------
    {
        Intrinsics in{};
        ERROR_CODE r = camera->getIntrinsics(STREAM_TYPE_DEPTH, in);
        bool ok = check("getIntrinsics(depth)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.intrinsics_depth.ok = ok;
        if (ok) {
            g_state.intrinsics_depth.width = in.width; g_state.intrinsics_depth.height = in.height;
            g_state.intrinsics_depth.fx = in.fx; g_state.intrinsics_depth.zero01 = in.zero01;
            g_state.intrinsics_depth.cx = in.cx; g_state.intrinsics_depth.zeor10 = in.zeor10;
            g_state.intrinsics_depth.fy = in.fy; g_state.intrinsics_depth.cy = in.cy;
            g_state.intrinsics_depth.zeor20 = in.zeor20; g_state.intrinsics_depth.zero21 = in.zero21;
            g_state.intrinsics_depth.one22 = in.one22;
        }
    }
    {
        Intrinsics in{};
        ERROR_CODE r = camera->getIntrinsics(STREAM_TYPE_RGB, in);
        bool ok = check("getIntrinsics(rgb)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.intrinsics_rgb.ok = ok;
        if (ok) {
            g_state.intrinsics_rgb.width = in.width; g_state.intrinsics_rgb.height = in.height;
            g_state.intrinsics_rgb.fx = in.fx; g_state.intrinsics_rgb.zero01 = in.zero01;
            g_state.intrinsics_rgb.cx = in.cx; g_state.intrinsics_rgb.zeor10 = in.zeor10;
            g_state.intrinsics_rgb.fy = in.fy; g_state.intrinsics_rgb.cy = in.cy;
            g_state.intrinsics_rgb.zeor20 = in.zeor20; g_state.intrinsics_rgb.zero21 = in.zero21;
            g_state.intrinsics_rgb.one22 = in.one22;
        }
    }
    {
        Extrinsics ex{};
        ERROR_CODE r = camera->getExtrinsics(ex);
        bool ok = check("getExtrinsics", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.extrinsics_ok = ok;
        if (ok) {
            for (int i = 0; i < 9; i++) g_state.extrinsics_rotation[i] = ex.rotation[i];
            for (int i = 0; i < 3; i++) g_state.extrinsics_translation[i] = ex.translation[i];
        }
    }
    {
        Distort d{};
        ERROR_CODE r = camera->getDistort(STREAM_TYPE_DEPTH, d);
        bool ok = check("getDistort(depth)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.distort_depth.ok = ok;
        if (ok) { g_state.distort_depth.k1=d.k1; g_state.distort_depth.k2=d.k2; g_state.distort_depth.k3=d.k3; g_state.distort_depth.k4=d.k4; g_state.distort_depth.k5=d.k5; }
    }
    {
        Distort d{};
        ERROR_CODE r = camera->getDistort(STREAM_TYPE_RGB, d);
        bool ok = check("getDistort(rgb)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.distort_rgb.ok = ok;
        if (ok) { g_state.distort_rgb.k1=d.k1; g_state.distort_rgb.k2=d.k2; g_state.distort_rgb.k3=d.k3; g_state.distort_rgb.k4=d.k4; g_state.distort_rgb.k5=d.k5; }
    }
    {
        PropertyExtension pe{};
        ERROR_CODE r = camera->getPropertyExtension(PROPERTY_EXT_DEPTH_SCALE, pe);
        bool ok = check("getPropertyExtension(DEPTH_SCALE)", r);
        std::lock_guard<std::mutex> lk(g_state.mu);
        g_state.depth_scale_ok = ok;
        if (ok) g_state.depth_scale = pe.depthScale;
    }

    // --- start streams + capture frames -------------------------------------
    if (haveDepthChoice) {
        ERROR_CODE r = camera->startStream(STREAM_TYPE_DEPTH, depthChoice);
        bool ok = check("startStream(depth)", r);
        depth_started = ok;
    }
    if (haveRgbChoice) {
        ERROR_CODE r = camera->startStream(STREAM_TYPE_RGB, rgbChoice);
        bool ok = check("startStream(rgb)", r);
        rgb_started = ok;
    }

    if (depth_started && rgb_started) {
        for (long i = 0; i < frames_requested; i++) {
            long rem = remaining_ms();
            if (rem < 300) {
                g_state.add_error("getPairedFrame", ERROR_FRAME_TIMEOUT, "aborting frame loop: insufficient time budget remaining");
                break;
            }
            long frames_left = frames_requested - i;
            long per_call = std::min<long>(3000, rem / frames_left);
            if (per_call < 200) per_call = 200;

            IFramePtr depthFrame, rgbFrame;
            ERROR_CODE r = camera->getPairedFrame(depthFrame, rgbFrame, (int)per_call);
            if (!check("getPairedFrame", r)) {
                continue;
            }
            if (depthFrame && !depthFrame->empty()) {
                std::lock_guard<std::mutex> lk(g_state.mu);
                auto &f = g_state.depth_frames;
                double ts = depthFrame->getTimeStamp();
                if (f.count == 0) { f.first_ts = ts; f.have_ts = true; }
                f.last_ts = ts;
                f.count++;
                f.width = depthFrame->getWidth();
                f.height = depthFrame->getHeight();
                f.format = (int)depthFrame->getFormat();
                f.format_name = stream_format_name(f.format);
                f.size_bytes = depthFrame->getSize();
            }
            if (rgbFrame && !rgbFrame->empty()) {
                std::lock_guard<std::mutex> lk(g_state.mu);
                auto &f = g_state.rgb_frames;
                double ts = rgbFrame->getTimeStamp();
                if (f.count == 0) { f.first_ts = ts; f.have_ts = true; }
                f.last_ts = ts;
                f.count++;
                f.width = rgbFrame->getWidth();
                f.height = rgbFrame->getHeight();
                f.format = (int)rgbFrame->getFormat();
                f.format_name = stream_format_name(f.format);
                f.size_bytes = rgbFrame->getSize();
            }
        }
    } else {
        g_state.add_error("startStream", ERROR_STREAM_NOT_START,
                           "one or both streams failed to start; skipping frame capture");
    }

    bool got_both = (g_state.depth_frames.count > 0 && g_state.rgb_frames.count > 0);
    std::string status = got_both ? "ok" : "no_frames";
    int code = got_both ? 0 : 4;
    return finish(status, code);
}
