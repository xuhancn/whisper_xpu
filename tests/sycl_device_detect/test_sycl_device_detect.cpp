#include <gtest/gtest.h>
#include "device_detect.h"

// ============================================================================
// Constants
// ============================================================================

TEST(Constants, kDeviceCPU) {
    EXPECT_EQ(kDeviceCPU, -1);
}

TEST(Constants, kDeviceAuto) {
    EXPECT_EQ(kDeviceAuto, -2);
}

// ============================================================================
// DeviceClass
// ============================================================================

TEST(DeviceClass, ValuesAreDistinct) {
    EXPECT_NE(DeviceClass::CPU, DeviceClass::GPU_Discrete);
    EXPECT_NE(DeviceClass::GPU_Integrated, DeviceClass::GPU_Discrete);
    EXPECT_NE(DeviceClass::GPU_Integrated, DeviceClass::Unknown);
}

// ============================================================================
// DeviceInfo::to_string()
// ============================================================================

TEST(DeviceInfoToString, CPU) {
    DeviceInfo cpu;
    cpu.index         = kDeviceCPU;
    cpu.device_class  = DeviceClass::CPU;
    cpu.name          = "CPU";
    cpu.vendor        = "CPU";
    cpu.compute_units = 8;
    cpu.total_mem     = 0;
    cpu.free_mem      = 0;

    std::string s = cpu.to_string();
    EXPECT_NE(s.find("[CPU]"), std::string::npos) << "should contain [CPU] tag";
    EXPECT_EQ(s.find("VRAM"), std::string::npos) << "CPU should have no VRAM";
}

TEST(DeviceInfoToString, dGPU) {
    DeviceInfo gpu;
    gpu.index         = 0;
    gpu.device_class  = DeviceClass::GPU_Discrete;
    gpu.name          = "Intel(R) Arc(TM) A770 Graphics";
    gpu.vendor        = "Intel(R) Corporation";
    gpu.compute_units = 512;
    gpu.total_mem     = 17163091968ULL;
    gpu.free_mem      = 8581545984ULL;

    std::string s = gpu.to_string();
    EXPECT_NE(s.find("[dGPU]"), std::string::npos) << "should contain [dGPU] tag";
    EXPECT_NE(s.find("VRAM"), std::string::npos) << "should contain VRAM";
    EXPECT_NE(s.find("free"), std::string::npos) << "should contain free memory";
    EXPECT_NE(s.find("512 CUs"), std::string::npos) << "should contain compute units";
}

TEST(DeviceInfoToString, iGPU) {
    DeviceInfo igpu;
    igpu.index         = 1;
    igpu.device_class  = DeviceClass::GPU_Integrated;
    igpu.name          = "Intel(R) UHD Graphics";
    igpu.vendor        = "Intel(R) Corporation";
    igpu.compute_units = 96;
    igpu.total_mem     = 34359738368ULL;
    igpu.free_mem      = 0;

    std::string s = igpu.to_string();
    EXPECT_NE(s.find("[iGPU]"), std::string::npos) << "should contain [iGPU] tag";
    EXPECT_NE(s.find("UHD Graphics"), std::string::npos) << "should contain device name";
    EXPECT_NE(s.find("Intel"), std::string::npos) << "should contain vendor";
}

TEST(DeviceInfoToString, UnknownClass) {
    DeviceInfo unk;
    unk.index         = 3;
    unk.device_class  = DeviceClass::Unknown;
    unk.name          = "Some Device";
    unk.vendor        = "Some Vendor";
    unk.compute_units = 0;
    unk.total_mem     = 0;
    unk.free_mem      = 0;

    std::string s = unk.to_string();
    EXPECT_NE(s.find("[Unknown]"), std::string::npos) << "should contain [Unknown] tag";
}

// ============================================================================
// get_device_info()
// ============================================================================

TEST(GetDeviceInfo, FallbackToCPU) {
    auto info = get_device_info(999);
    EXPECT_EQ(info.index, kDeviceCPU);
    EXPECT_EQ(info.device_class, DeviceClass::CPU);
}

TEST(GetDeviceInfo, ReturnsCPUEntry) {
    auto info = get_device_info(kDeviceCPU);
    EXPECT_EQ(info.index, kDeviceCPU);
    EXPECT_EQ(info.device_class, DeviceClass::CPU);
}

// ============================================================================
// SYCL-dependent (only compiled when WHISPER_XPU_HAS_SYCL is defined)
// ============================================================================

#ifdef WHISPER_XPU_HAS_SYCL
TEST(SYCL, HasIntelGPU) {
    EXPECT_NO_THROW({
        bool has = has_intel_gpu();
        EXPECT_TRUE(has) << "expected at least one Intel GPU";
    });
}

TEST(SYCL, GetAvailableDevices) {
    auto devices = get_available_devices();
    ASSERT_FALSE(devices.empty()) << "device list should not be empty";
    EXPECT_EQ(devices[0].index, kDeviceCPU);
    EXPECT_EQ(devices[0].device_class, DeviceClass::CPU);

    for (const auto& d : devices) {
        if (d.device_class != DeviceClass::CPU) {
            EXPECT_GE(d.index, 0) << "GPU index must be >= 0";
        }
    }
}
#endif
