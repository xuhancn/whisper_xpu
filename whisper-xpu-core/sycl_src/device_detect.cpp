#include "device_detect.h"

#ifdef WHISPER_XPU_HAS_SYCL
// Enumerate Intel SYCL GPUs via the ggml SYCL backend API (declared in
// ggml-sycl.h), NOT via <sycl/sycl.hpp>.  Including <sycl/sycl.hpp> under icpx
// on Windows trips MSVC STL's _Get_function_impl static_assert for the
// std::function<> types SYCL uses internally (async_handler, device_selector,
// property_list) — "std::function only accepts function types as template
// arguments".  The ggml API (ggml_backend_sycl_get_device_count/description/
// memory) is already compiled into this same DLL via /WHOLEARCHIVE:ggml-sycl,
// gives us device count + name + VRAM, and is the same path probe_sycl_device
// (engine.cpp) uses successfully at runtime.  We lose max_compute_units and
// host_unified_memory vs the raw SYCL path, but classify iGPU/dGPU by name.
#include "ggml-sycl.h"
#endif
#include <sstream>
#include <algorithm>
#include <thread>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>

namespace whisper_xpu {

// ---------------------------------------------------------------------------
static std::string device_class_name(DeviceClass dc) {
    switch (dc) {
        case DeviceClass::CPU: return "CPU";
        case DeviceClass::GPU_Integrated: return "iGPU";
        case DeviceClass::GPU_Discrete: return "dGPU";
        default: return "Unknown";
    }
}

#ifdef WHISPER_XPU_HAS_SYCL
// Classify iGPU vs dGPU by name.  Intel integrated parts are marketed as
// UHD / Iris / Iris Plus / Iris Pro / Iris Xe / "Graphics"; discrete Arc
// cards are "Arc".  This mirrors the name-fallback the raw-SYCL path used
// when host_unified_memory() threw.
static DeviceClass classify_by_name(const std::string& name) {
    // Case-insensitive substring search via a lowercased copy.
    std::string n;
    n.reserve(name.size());
    for (char c : name) n.push_back((char)std::tolower((unsigned char)c));
    auto has = [&](const char* k) { return n.find(k) != std::string::npos; };
    if (has("arc")) return DeviceClass::GPU_Discrete;
    if (has("uhd") || has("iris") || has("xe") || has("graphics"))
        return DeviceClass::GPU_Integrated;
    // Unknown Intel GPU — assume discrete (safer default for large-VRAM parts).
    return DeviceClass::GPU_Discrete;
}

// SEH crash-guard for the ggml SYCL calls is kept in a SEPARATE function that
// holds no C++ object with a destructor.  MSVC rejects __try in functions that
// require object unwinding (C2712); a function touching only POD is allowed.
// The ggml SYCL calls reach into the SYCL runtime / Level Zero loader, which
// can access-violate on a broken driver/oneAPI combo — __except(1) catches the
// AV and we fall back to CPU-only, matching prior behavior.
// (icpx tolerated __try alongside C++ objects; the core is now MSVC-compiled,
//  so this split is mandatory.)
struct RawSyclDevice {
    int    index;
    char   name[256];
    size_t total_mem;
    size_t free_mem;
};
static int enum_sycl_devices_raw(RawSyclDevice* out, int max_count) {
    __try {
        int n = ggml_backend_sycl_get_device_count();
        if (n > max_count) n = max_count;
        for (int dev = 0; dev < n; ++dev) {
            out[dev].index     = dev;
            out[dev].name[0]   = '\0';
            out[dev].total_mem = 0;
            out[dev].free_mem  = 0;
            ggml_backend_sycl_get_device_description(dev, out[dev].name,
                                                     sizeof(out[dev].name));
            ggml_backend_sycl_get_device_memory(dev, &out[dev].free_mem,
                                                &out[dev].total_mem);
        }
        return n;
    } __except(1) {
        // SYCL runtime / Level Zero crashed — report no GPUs.
        return 0;
    }
}

// Init-level usability probe, SEH-guarded (POD-only, like enum_sycl_devices_raw).
// Constructs + immediately frees a SYCL backend for `dev`.  Returns false if
// the SYCL runtime / Level Zero driver access-violates during backend setup.
// HONEST LIMIT: this catches driver faults at BACKEND-INIT time only.  A
// future COMPUTE-level regression (e.g. the old urProgramBuildExp AV that
// rejected `-ze-intel-greater-than-4GB-buffer-required`) happens later, during
// kernel JIT/warmup, and is NOT caught here — it would still hard-crash at
// load time.  The UI greys (but does NOT hide) devices this returns false for,
// so the user sees the device exists but can't pick it.  See memory
// [[igpu-iris-xe-not-usable]] / [[ze-driver-crash-under-debugger]].
static bool probe_device_usable_raw(int dev) {
    __try {
        ggml_backend_t backend = ggml_backend_sycl_init(dev);
        if (!backend) return false;
        ggml_backend_free(backend);
        return true;
    } __except(1) {
        return false;
    }
}
#endif

// ---------------------------------------------------------------------------
WHISPER_XPU_SYCL_API std::vector<DeviceInfo> get_available_devices() {
    std::vector<DeviceInfo> list;

    // CPU is always available
    DeviceInfo cpu;
    cpu.index = kDeviceCPU; cpu.device_class = DeviceClass::CPU;
    cpu.name = "CPU"; cpu.vendor = "CPU";
    cpu.compute_units = (int)std::thread::hardware_concurrency();
    cpu.total_mem = 0; cpu.free_mem = 0;
    list.push_back(cpu);

#ifdef WHISPER_XPU_HAS_SYCL
    // Enumerate GPUs via the SEH-guarded POD helper (see above), then promote
    // the raw results to DeviceInfo.  classify_by_name() builds std::strings,
    // so it stays out of the __try function.
    RawSyclDevice raw[32];
    int n = enum_sycl_devices_raw(raw, 32);
    for (int dev = 0; dev < n; ++dev) {
        DeviceInfo inf;
        inf.index         = raw[dev].index;
        inf.device_class  = classify_by_name(raw[dev].name);
        inf.name          = raw[dev].name;
        inf.vendor        = "Intel";  // ggml SYCL backend targets Intel
        inf.compute_units = 0;        // not exposed by the ggml API
        inf.total_mem     = raw[dev].total_mem;
        inf.free_mem      = raw[dev].free_mem;
        // SEH-guarded init probe: usable unless the driver AVs at backend setup.
        inf.usable        = probe_device_usable_raw(dev);
        list.push_back(inf);
    }
#endif
    return list;
}

// ---------------------------------------------------------------------------
WHISPER_XPU_SYCL_API bool has_intel_gpu() {
#ifdef WHISPER_XPU_HAS_SYCL
    __try {
        return ggml_backend_sycl_get_device_count() > 0;
    } __except(1) {
        return false;
    }
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
WHISPER_XPU_SYCL_API DeviceInfo get_device_info(int i) {
    auto d = get_available_devices();
    for (auto &x : d) if (x.index == i) return x;
    for (auto &x : d) if (x.index == kDeviceCPU) return x;
    DeviceInfo f; f.index = kDeviceCPU; f.device_class = DeviceClass::CPU; f.name = "CPU"; return f;
}

std::string DeviceInfo::to_string() const {
    std::ostringstream os;
    os << "[" << device_class_name(device_class) << "] " << name;
    if (vendor != "CPU" && !vendor.empty()) os << " (" << vendor << ")";
    if (total_mem > 0) {
        os << " | VRAM: " << (total_mem / (1024*1024)) << " MB";
        if (free_mem > 0) os << " (free: " << (free_mem / (1024*1024)) << " MB)";
    }
    if (compute_units > 0) os << " | " << compute_units << " CUs";
    return os.str();
}

} // namespace whisper_xpu
