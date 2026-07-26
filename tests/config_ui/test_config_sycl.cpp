#include <gtest/gtest.h>
#include "device_detect.h"
#include <string>
#include <vector>

// ============================================================================
// DeviceInfo::to_string() — no SYCL runtime needed
// ============================================================================

TEST(DeviceInfo, CPUToString) {
    whisper_xpu::DeviceInfo d;
    d.index         = kDeviceCPU;
    d.device_class  = DeviceClass::CPU;
    d.name          = "Test CPU";
    d.vendor        = "ACME";
    d.compute_units = 8;
    d.total_mem     = 0;
    d.free_mem      = 0;

    std::string s = d.to_string();
    EXPECT_NE(s.find("[CPU]"), std::string::npos);
    EXPECT_NE(s.find("Test CPU"), std::string::npos);
    EXPECT_EQ(s.find("VRAM"), std::string::npos);
}

TEST(DeviceInfo, dGPUToString) {
    whisper_xpu::DeviceInfo d;
    d.index         = 0;
    d.device_class  = DeviceClass::GPU_Discrete;
    d.name          = "Arc A770";
    d.vendor        = "Intel";
    d.compute_units = 512;
    d.total_mem     = 16ULL * 1024 * 1024 * 1024;
    d.free_mem      = 8ULL * 1024 * 1024 * 1024;

    std::string s = d.to_string();
    EXPECT_NE(s.find("[dGPU]"), std::string::npos);
    EXPECT_NE(s.find("VRAM"), std::string::npos);
    EXPECT_NE(s.find("free"), std::string::npos);
    EXPECT_NE(s.find("512 CUs"), std::string::npos);
}

TEST(DeviceInfo, iGPUToString) {
    whisper_xpu::DeviceInfo d;
    d.index         = 1;
    d.device_class  = DeviceClass::GPU_Integrated;
    d.name          = "UHD Graphics";
    d.vendor        = "Intel";
    d.compute_units = 96;
    d.total_mem     = 32ULL * 1024 * 1024 * 1024;
    d.free_mem      = 0;

    std::string s = d.to_string();
    EXPECT_NE(s.find("[iGPU]"), std::string::npos);
    EXPECT_NE(s.find("UHD Graphics"), std::string::npos);
}

TEST(DeviceInfo, UnknownToString) {
    whisper_xpu::DeviceInfo d;
    d.index         = 9;
    d.device_class  = DeviceClass::Unknown;
    d.name          = "Mystery Device";
    d.vendor        = "???";
    d.compute_units = 0;
    d.total_mem     = 0;
    d.free_mem      = 0;

    std::string s = d.to_string();
    EXPECT_NE(s.find("[Unknown]"), std::string::npos);
}

// ============================================================================
// get_device_info() — queries SYCL runtime
// ============================================================================

TEST(GetDeviceInfo, BadIndexFallsBackToCPU) {
    auto info = whisper_xpu::get_device_info(99999);
    EXPECT_EQ(info.index, kDeviceCPU);
    EXPECT_EQ(info.device_class, DeviceClass::CPU);
}

TEST(GetDeviceInfo, CPUExplicit) {
    auto info = whisper_xpu::get_device_info(kDeviceCPU);
    EXPECT_EQ(info.index, kDeviceCPU);
    EXPECT_EQ(info.device_class, DeviceClass::CPU);
}

// ============================================================================
// get_available_devices() — full SYCL enumeration
// ============================================================================

TEST(GetAvailableDevices, ReturnsNonEmpty) {
    auto devices = whisper_xpu::get_available_devices();
    ASSERT_FALSE(devices.empty());
}

TEST(GetAvailableDevices, FirstEntryIsCPU) {
    auto devices = whisper_xpu::get_available_devices();
    ASSERT_FALSE(devices.empty());
    EXPECT_EQ(devices[0].index, kDeviceCPU);
    EXPECT_EQ(devices[0].device_class, DeviceClass::CPU);
}

TEST(GetAvailableDevices, GPUIndicesAreNonNegative) {
    auto devices = whisper_xpu::get_available_devices();
    for (const auto& d : devices) {
        if (d.device_class != DeviceClass::CPU) {
            EXPECT_GE(d.index, 0) << d.name;
        }
    }
}

TEST(GetAvailableDevices, AllNamesAndVendorsPresent) {
    auto devices = whisper_xpu::get_available_devices();
    for (const auto& d : devices) {
        EXPECT_FALSE(d.name.empty());
        EXPECT_FALSE(d.vendor.empty());
    }
}

TEST(GetAvailableDevices, GPUReportsComputeUnits) {
    auto devices = whisper_xpu::get_available_devices();
    bool found_gpu = false;
    for (const auto& d : devices) {
        if (d.device_class == DeviceClass::GPU_Discrete ||
            d.device_class == DeviceClass::GPU_Integrated) {
            found_gpu = true;
            EXPECT_GT(d.compute_units, 0) << d.name << " should report CUs > 0";
        }
    }
    // Not asserting found_gpu == true — system may have no Intel GPU.
    // Test passes either way; the assertion above runs only if a GPU exists.
    (void)found_gpu;
}

TEST(GetAvailableDevices, IntGPUFunctions) {
    // Just verify these don't crash
    EXPECT_NO_THROW({
        bool has = whisper_xpu::has_intel_gpu();
        (void)has;
    });
}
