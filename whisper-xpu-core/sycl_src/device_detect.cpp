#include "device_detect.h"

#ifdef WHISPER_XPU_HAS_SYCL
#include <sycl/sycl.hpp>
#include "ggml-sycl.h"
#endif

#include <sstream>
#include <algorithm>
#include <thread>
#include <cstring>

// ---------------------------------------------------------------------------
// helpers (SYCL-free)
// ---------------------------------------------------------------------------

static std::string device_class_name(DeviceClass dc) {
    switch (dc) {
        case DeviceClass::CPU:             return "CPU";
        case DeviceClass::GPU_Integrated:  return "iGPU";
        case DeviceClass::GPU_Discrete:    return "dGPU";
        default:                           return "Unknown";
    }
}

#ifdef WHISPER_XPU_HAS_SYCL

static DeviceClass classify_device(const sycl::device &dev) {
    if (dev.is_cpu())
        return DeviceClass::CPU;
    if (!dev.is_gpu())
        return DeviceClass::Unknown;
    bool integrated = false;
    try {
        integrated = dev.get_info<sycl::info::device::host_unified_memory>();
    } catch (...) {
        std::string name = dev.get_info<sycl::info::device::name>();
        if (name.find("Graphics") != std::string::npos ||
            name.find("UHD") != std::string::npos ||
            name.find("Iris") != std::string::npos)
            integrated = true;
        else
            integrated = false;
    }
    return integrated ? DeviceClass::GPU_Integrated : DeviceClass::GPU_Discrete;
}

#endif // WHISPER_XPU_HAS_SYCL

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

std::vector<DeviceInfo> get_available_devices() {
    std::vector<DeviceInfo> list;

    // ── CPU (always first) ──
    {
        DeviceInfo cpu;
        cpu.index         = kDeviceCPU;
        cpu.device_class  = DeviceClass::CPU;
        cpu.name          = "CPU";
        cpu.vendor        = "CPU";
        cpu.compute_units = static_cast<int>(std::thread::hardware_concurrency());
        cpu.total_mem     = 0;
        cpu.free_mem      = 0;
        list.push_back(cpu);
    }

#ifdef WHISPER_XPU_HAS_SYCL
    try {
        int sycl_device_idx = 0;
        auto platforms = sycl::platform::get_platforms();
        for (const auto &platform : platforms) {
            auto devices = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto &dev : devices) {
                DeviceInfo info;
                info.index        = sycl_device_idx;
                info.device_class = classify_device(dev);
                info.name         = dev.get_info<sycl::info::device::name>();
                info.vendor       = dev.get_info<sycl::info::device::vendor>();
                info.compute_units = static_cast<int>(
                    dev.get_info<sycl::info::device::max_compute_units>());
                info.total_mem = dev.get_info<sycl::info::device::global_mem_size>();

                size_t free_s = 0, total_s = 0;
                ggml_backend_sycl_get_device_memory(sycl_device_idx, &free_s, &total_s);
                info.free_mem = free_s;

                list.push_back(info);
                ++sycl_device_idx;
            }
        }
    } catch (const std::exception &e) {
        DeviceInfo err;
        err.index        = -2;
        err.device_class = DeviceClass::Unknown;
        err.name         = "SYCL error: " + std::string(e.what());
        list.push_back(err);
    }
#else
    DeviceInfo sycl_disabled;
    sycl_disabled.index        = -2;
    sycl_disabled.device_class = DeviceClass::Unknown;
    sycl_disabled.name         = "SYCL support not compiled in";
    list.push_back(sycl_disabled);
#endif

    return list;
}

bool has_intel_gpu() {
#ifdef WHISPER_XPU_HAS_SYCL
    try {
        for (const auto &platform : sycl::platform::get_platforms()) {
            auto gpus = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto &gpu : gpus) {
                auto vendor = gpu.get_info<sycl::info::device::vendor>();
                if (vendor.find("Intel") != std::string::npos) {
                    return true;
                }
            }
        }
    } catch (...) {}
#endif
    return false;
}

DeviceInfo get_device_info(int device_index) {
    auto devices = get_available_devices();
    for (const auto &d : devices) {
        if (d.index == device_index) return d;
    }
    // If running on a system with no GPU, devices may only have CPU.
    // But if none matched at all (including CPU), return a fallback.
    for (const auto &d : devices) {
        if (d.index == kDeviceCPU) return d;
    }
    // Degenerate: no devices at all
    DeviceInfo fallback;
    fallback.index        = kDeviceCPU;
    fallback.device_class = DeviceClass::CPU;
    fallback.name         = "CPU";
    return fallback;
}

std::string DeviceInfo::to_string() const {
    std::ostringstream oss;
    oss << "[" << device_class_name(device_class) << "] " << name;
    if (vendor != "CPU" && !vendor.empty())
        oss << " (" << vendor << ")";
    if (total_mem > 0) {
        oss << " | VRAM: " << (total_mem / (1024 * 1024)) << " MB";
        if (free_mem > 0)
            oss << " (free: " << (free_mem / (1024 * 1024)) << " MB)";
    }
    if (compute_units > 0)
        oss << " | " << compute_units << " CUs";
    return oss.str();
}
