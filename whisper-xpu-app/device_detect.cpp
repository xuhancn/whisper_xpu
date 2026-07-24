#include "device_detect.h"

#ifdef WHISPER_XPU_HAS_SYCL
#include <sycl/sycl.hpp>
#endif

#include <sstream>
#include <algorithm>
#include <thread>

std::vector<std::string> detect_devices() {
    std::vector<std::string> devices;

#ifdef WHISPER_XPU_HAS_SYCL
    try {
        for (const auto& platform : sycl::platform::get_platforms()) {
            auto all_devices = platform.get_devices();
            for (const auto& device : all_devices) {
                std::ostringstream oss;
                oss << device.get_info<sycl::info::device::name>()
                    << " [" << device.get_info<sycl::info::device::vendor>()
                    << "] - "
                    << (device.is_gpu() ? "GPU" :
                        device.is_cpu() ? "CPU" : "Accelerator");

                auto version = device.get_info<sycl::info::device::version>();
                auto global_mem = device.get_info<sycl::info::device::global_mem_size>();

                oss << " | Version: " << version
                    << " | VRAM: " << (global_mem / (1024 * 1024)) << " MB";

                devices.push_back(oss.str());
            }
        }
    } catch (const sycl::exception& e) {
        devices.push_back("SYCL enumeration failed: " + std::string(e.what()));
    } catch (const std::exception& e) {
        devices.push_back("SYCL enumeration error: " + std::string(e.what()));
    }
#else
    devices.push_back("SYCL not compiled (use icpx compiler for GPU support)");
#endif

    // Always report CPU fallback
    {
        std::ostringstream oss;
        oss << "CPU (" << std::thread::hardware_concurrency() << " threads)";
        devices.push_back(oss.str());
    }

    return devices;
}

bool has_intel_gpu() {
#ifdef WHISPER_XPU_HAS_SYCL
    try {
        for (const auto& platform : sycl::platform::get_platforms()) {
            auto gpus = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto& gpu : gpus) {
                auto vendor = gpu.get_info<sycl::info::device::vendor>();
                if (vendor.find("Intel") != std::string::npos) {
                    return true;
                }
            }
        }
    } catch (...) { }
#endif
    return false;
}

std::string get_best_device_name() {
#ifdef WHISPER_XPU_HAS_SYCL
    try {
        for (const auto& platform : sycl::platform::get_platforms()) {
            auto gpus = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto& gpu : gpus) {
                auto vendor = gpu.get_info<sycl::info::device::vendor>();
                if (vendor.find("Intel") != std::string::npos) {
                    auto mem = gpu.get_info<sycl::info::device::global_mem_size>();
                    return gpu.get_info<sycl::info::device::name>() + " (Intel GPU, " +
                           std::to_string(mem / (1024 * 1024)) + " MB)";
                }
            }
        }
    } catch (...) { }
#endif
    return "CPU (" + std::to_string(std::thread::hardware_concurrency()) + " threads)";
}
