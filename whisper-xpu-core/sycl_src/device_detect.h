#pragma once

#include "export.h"
#include <string>
#include <vector>

// Device index constants
static constexpr int kDeviceCPU  = -1;
static constexpr int kDeviceAuto = -2;

enum class DeviceClass { CPU, GPU_Integrated, GPU_Discrete, Unknown };

namespace whisper_xpu {

struct WHISPER_XPU_SYCL_API DeviceInfo {
    int         index;
    DeviceClass device_class;
    std::string name;
    std::string vendor;
    int         compute_units;
    size_t      total_mem;
    size_t      free_mem;
    // True if a SEH-guarded ggml_backend_sycl_init succeeded on this device
    // (an init-level probe — catches driver faults at backend setup; a
    // compute-level regression like the old urProgramBuildExp AV would still
    // crash at load time and is NOT caught here).  The UI greys out (but does
    // NOT hide) devices with usable=false so the user sees the device exists.
    bool        usable = true;
    std::string to_string() const;
};

WHISPER_XPU_SYCL_API std::vector<DeviceInfo> get_available_devices();
WHISPER_XPU_SYCL_API bool has_intel_gpu();
WHISPER_XPU_SYCL_API DeviceInfo get_device_info(int device_index);

} // namespace whisper_xpu
