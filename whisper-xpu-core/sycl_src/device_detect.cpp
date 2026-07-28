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
    // SEH guard: the ggml SYCL calls reach into sycl8.dll / the Level Zero
    // loader, which can AV on broken driver/oneAPI combos.  icpx supports
    // __try with C++ objects (MSVC C2712 doesn't).  __except(1) catches only
    // the AV — on crash we return CPU-only, matching prior behavior.
    __try {
        int n = ggml_backend_sycl_get_device_count();
        for (int dev = 0; dev < n; ++dev) {
            char desc[256] = {0};
            ggml_backend_sycl_get_device_description(dev, desc, sizeof(desc));
            size_t free_mem = 0, total_mem = 0;
            ggml_backend_sycl_get_device_memory(dev, &free_mem, &total_mem);

            DeviceInfo inf;
            inf.index         = dev;
            inf.device_class  = classify_by_name(desc);
            inf.name          = desc;
            inf.vendor        = "Intel";  // ggml SYCL backend targets Intel
            inf.compute_units = 0;        // not exposed by the ggml API
            inf.total_mem     = total_mem;
            inf.free_mem      = free_mem;
            list.push_back(inf);
        }
    } __except(1) {
        // sycl8.dll / Level Zero crashed — return CPU-only.
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
