#pragma once

#include "export.h"
#include <string>
#include <vector>

// Device index constants
static constexpr int kDeviceCPU  = -1;
static constexpr int kDeviceAuto = -2;

enum class DeviceClass { CPU, GPU_Integrated, GPU_Discrete, Unknown };

namespace whisper_xpu {

struct WHISPER_XPU_API DeviceInfo {
    int         index;
    DeviceClass device_class;
    std::string name;
    std::string vendor;
    int         compute_units;
    size_t      total_mem;
    size_t      free_mem;
    std::string to_string() const;
};

std::vector<DeviceInfo> get_available_devices();
bool has_intel_gpu();
DeviceInfo get_device_info(int device_index);

} // namespace whisper_xpu
