#include "device_detect.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <sstream>

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name)                                                       \
    do {                                                                 \
        printf("  TEST: %s ... ", name);                                 \
        fflush(stdout);                                                  \
    } while (0)

#define PASS()                                                           \
    do {                                                                 \
        printf("PASSED\n");                                              \
        g_tests_passed++;                                                \
    } while (0)

#define FAIL(msg)                                                        \
    do {                                                                 \
        printf("FAILED: %s\n", msg);                                     \
        g_tests_failed++;                                                \
    } while (0)

#define ASSERT(cond, msg)                                                \
    do {                                                                 \
        if (!(cond)) { FAIL(msg); return; }                              \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static void test_constants() {
    TEST("kDeviceCPU == -1");
    ASSERT(kDeviceCPU == -1, "expected -1");
    PASS();

    TEST("kDeviceAuto == -2");
    ASSERT(kDeviceAuto == -2, "expected -2");
    PASS();
}

// ---------------------------------------------------------------------------
// DeviceClass
// ---------------------------------------------------------------------------

static void test_device_class() {
    TEST("DeviceClass::CPU != DeviceClass::GPU_Discrete");
    ASSERT(DeviceClass::CPU != DeviceClass::GPU_Discrete, "expected different values");
    PASS();

    TEST("DeviceClass::GPU_Integrated != DeviceClass::GPU_Discrete");
    ASSERT(DeviceClass::GPU_Integrated != DeviceClass::GPU_Discrete, "expected different values");
    PASS();

    TEST("DeviceClass::GPU_Integrated != DeviceClass::Unknown");
    ASSERT(DeviceClass::GPU_Integrated != DeviceClass::Unknown, "expected different values");
    PASS();
}

// ---------------------------------------------------------------------------
// DeviceInfo::to_string()
// ---------------------------------------------------------------------------

static void test_device_info_to_string_cpu() {
    DeviceInfo cpu;
    cpu.index         = kDeviceCPU;
    cpu.device_class  = DeviceClass::CPU;
    cpu.name          = "CPU";
    cpu.vendor        = "CPU";
    cpu.compute_units = 8;
    cpu.total_mem     = 0;
    cpu.free_mem      = 0;

    std::string s = cpu.to_string();
    TEST("CPU to_string contains [CPU]");
    ASSERT(s.find("[CPU]") != std::string::npos, "expected [CPU] tag");
    PASS();

    TEST("CPU to_string does not contain VRAM");
    ASSERT(s.find("VRAM") == std::string::npos, "CPU should have no VRAM");
    PASS();
}

static void test_device_info_to_string_gpu() {
    DeviceInfo gpu;
    gpu.index         = 0;
    gpu.device_class  = DeviceClass::GPU_Discrete;
    gpu.name          = "Intel(R) Arc(TM) A770 Graphics";
    gpu.vendor        = "Intel(R) Corporation";
    gpu.compute_units = 512;
    gpu.total_mem     = 17163091968ULL;   // ~16 GB
    gpu.free_mem      = 8581545984ULL;    // ~8 GB

    std::string s = gpu.to_string();
    TEST("dGPU to_string contains [dGPU]");
    ASSERT(s.find("[dGPU]") != std::string::npos, "expected [dGPU] tag");
    PASS();

    TEST("dGPU to_string contains VRAM in MB");
    ASSERT(s.find("VRAM") != std::string::npos, "expected VRAM");
    PASS();

    TEST("dGPU to_string contains free mem");
    ASSERT(s.find("free") != std::string::npos, "expected free memory");
    PASS();

    TEST("dGPU to_string contains CU count");
    ASSERT(s.find("512 CUs") != std::string::npos, "expected compute units");
    PASS();
}

static void test_device_info_to_string_igpu() {
    DeviceInfo igpu;
    igpu.index         = 1;
    igpu.device_class  = DeviceClass::GPU_Integrated;
    igpu.name          = "Intel(R) UHD Graphics";
    igpu.vendor        = "Intel(R) Corporation";
    igpu.compute_units = 96;
    igpu.total_mem     = 34359738368ULL;  // 32 GB shared
    igpu.free_mem      = 0;               // free not available

    std::string s = igpu.to_string();
    TEST("iGPU to_string contains [iGPU]");
    ASSERT(s.find("[iGPU]") != std::string::npos, "expected [iGPU] tag");
    PASS();

    TEST("iGPU to_string mentions UHD Graphics");
    ASSERT(s.find("UHD Graphics") != std::string::npos, "expected device name");
    PASS();

    TEST("iGPU to_string has vendor");
    ASSERT(s.find("Intel") != std::string::npos, "expected vendor name");
    PASS();
}

// ---------------------------------------------------------------------------
// get_device_info() fallback
// ---------------------------------------------------------------------------

static void test_get_device_info_cpu_fallback() {
    // When no devices with the given index exist, should return CPU.
    auto info = get_device_info(999);
    TEST("get_device_info(999) returns CPU");
    ASSERT(info.index == kDeviceCPU, "expected CPU fallback");
    PASS();
}

// ---------------------------------------------------------------------------
// SYCL-dependent tests (only run when WHISPER_XPU_HAS_SYCL is defined)
// ---------------------------------------------------------------------------

#ifdef WHISPER_XPU_HAS_SYCL
static void test_has_intel_gpu() {
    TEST("has_intel_gpu() runs without exception");
    try {
        bool has = has_intel_gpu();
        printf(" (result: %s) ", has ? "true" : "false");
        PASS();
    } catch (const std::exception& e) {
        FAIL(e.what());
    }
}

static void test_get_available_devices() {
    TEST("get_available_devices() returns non-empty list");
    auto devices = get_available_devices();
    ASSERT(!devices.empty(), "device list should not be empty");
    PASS();

    TEST("get_available_devices() first entry is always CPU");
    ASSERT(devices[0].index == kDeviceCPU, "first entry must be CPU");
    ASSERT(devices[0].device_class == DeviceClass::CPU, "first entry must be CPU class");
    PASS();

    TEST("get_available_devices() GPU entries have positive index");
    for (const auto& d : devices) {
        if (d.device_class != DeviceClass::CPU) {
            ASSERT(d.index >= 0, "GPU index must be >= 0");
        }
    }
    PASS();
}
#else
static void test_sycl_not_available() {
    TEST("WHISPER_XPU_HAS_SYCL not defined — SYCL tests skipped");
    PASS();
}
#endif

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("=== whisper_xpu device_detect tests ===\n\n");

    printf("--- Constants ---\n");
    test_constants();

    printf("\n--- DeviceClass ---\n");
    test_device_class();

    printf("\n--- DeviceInfo::to_string() ---\n");
    test_device_info_to_string_cpu();
    test_device_info_to_string_gpu();
    test_device_info_to_string_igpu();

    printf("\n--- get_device_info() ---\n");
    test_get_device_info_cpu_fallback();

    printf("\n--- SYCL-dependent ---\n");
#ifdef WHISPER_XPU_HAS_SYCL
    test_has_intel_gpu();
    test_get_available_devices();
#else
    test_sycl_not_available();
#endif

    printf("\n=== Results: %d passed, %d failed ===\n",
           g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
