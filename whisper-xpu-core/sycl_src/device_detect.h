#include "export.h"
#pragma once

#include <string>
#include <vector>

// Device index constants
static constexpr int kDeviceCPU  = -1;   // CPU fallback
static constexpr int kDeviceAuto = -2;   // auto-detect best device

enum class DeviceClass { CPU, GPU_Integrated, GPU_Discrete, Unknown };

struct WHISPER_XPU_API DeviceInfo {
    int         index;          // device index for ggml_backend_sycl_init (kDeviceCPU = CPU)
    DeviceClass device_class;   // CPU / iGPU / dGPU
    std::string name;           // human-readable device name
    std::string vendor;         // e.g. "Intel(R) Corporation", "CPU"
    int         compute_units;  // max_compute_units (CPU = thread count)
    size_t      total_mem;      // bytes (0 for CPU)
    size_t      free_mem;       // bytes (0 for CPU, queried at enumeration time)
    std::string to_string() const;
};

// Returns list of all available devices. CPU is always first (index kDeviceCPU),
// followed by SYCL-enumerated GPUs (index 0, 1, ...).
std::vector<DeviceInfo> get_available_devices();

// Returns true if at least one Intel GPU is available via SYCL.
bool has_intel_gpu();

// Returns the device_info at the given global index,
// or the CPU default if index == kDeviceCPU.
DeviceInfo get_device_info(int device_index);
