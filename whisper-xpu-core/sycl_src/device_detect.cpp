#include "device_detect.h"

#ifdef WHISPER_XPU_HAS_SYCL
#include <sycl/sycl.hpp>
#include "ggml-sycl.h"
#endif
#include <sstream>
#include <algorithm>
#include <thread>
#include <cstring>

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
static DeviceClass classify_device(const sycl::device &dev) {
    if (dev.is_cpu() || !dev.is_gpu()) return dev.is_cpu() ? DeviceClass::CPU : DeviceClass::Unknown;
    bool integrated = false;
    try { integrated = dev.get_info<sycl::info::device::host_unified_memory>(); }
    catch (...) {
        auto n = dev.get_info<sycl::info::device::name>();
        integrated = (n.find("Graphics") != std::string::npos || n.find("UHD") != std::string::npos || n.find("Iris") != std::string::npos);
    }
    return integrated ? DeviceClass::GPU_Integrated : DeviceClass::GPU_Discrete;
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
    int gpu_count = 0;
    for (auto &p : sycl::platform::get_platforms()) {
        for (auto &d : p.get_devices(sycl::info::device_type::gpu)) {
            DeviceInfo inf;
            inf.index         = gpu_count;
            inf.device_class  = classify_device(d);
            inf.name          = d.get_info<sycl::info::device::name>();
            inf.vendor        = d.get_info<sycl::info::device::vendor>();
            inf.compute_units = (int)d.get_info<sycl::info::device::max_compute_units>();
            inf.total_mem     = d.get_info<sycl::info::device::global_mem_size>();
            inf.free_mem      = 0;
            list.push_back(inf);
            ++gpu_count;
        }
    }
#endif
    return list;
}

// ---------------------------------------------------------------------------
WHISPER_XPU_SYCL_API bool has_intel_gpu() {
#ifdef WHISPER_XPU_HAS_SYCL
    for (auto &p : sycl::platform::get_platforms())
        for (auto &g : p.get_devices(sycl::info::device_type::gpu))
            if (g.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos)
                return true;
#endif
    return false;
}

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
