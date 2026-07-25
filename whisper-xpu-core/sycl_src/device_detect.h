#pragma once

#include <string>
#include <vector>

enum class DeviceClass { CPU, GPU_Integrated, GPU_Discrete, Unknown };

struct DeviceInfo {
    int         index;          // device index for ggml_backend_sycl_init (-1 = CPU)
    DeviceClass device_class;   // CPU / iGPU / dGPU
    std::string name;           // human-readable device name
    std::string vendor;         // e.g. "Intel(R) Corporation", "CPU"
    int         compute_units;  // max_compute_units (CPU = thread count)
    size_t      total_mem;      // bytes (0 for CPU)
    size_t      free_mem;       // bytes (0 for CPU, queried at enumeration time)
    std::string to_string() const;
};

// Returns list of all available devices. CPU is always first (index -1),
// followed by SYCL-enumerated GPUs (index 0, 1, ...).
std::vector<DeviceInfo> get_available_devices();

// Returns true if at least one Intel GPU is available via SYCL.
bool has_intel_gpu();

// Returns the device_info at the given global index,
// or the CPU default if index == -1.
DeviceInfo get_device_info(int device_index);
