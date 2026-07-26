#include <gtest/gtest.h>
#include "audio_capture.h"
#include <string>

// ============================================================================
// Audio device enumeration (PortAudio, no SYCL needed)
// ============================================================================

TEST(AudioDevice, EnumerateReturnsNonEmpty) {
    auto devices = AudioCapture::enumerate_devices();
    EXPECT_FALSE(devices.empty())
        << "system should have at least one audio input device";
}

TEST(AudioDevice, AllEntriesHaveName) {
    auto devices = AudioCapture::enumerate_devices();
    for (const auto& d : devices) {
        EXPECT_FALSE(d.name.empty()) << "device index " << d.index;
    }
}

TEST(AudioDevice, MaxChannelsAreNonNegative) {
    auto devices = AudioCapture::enumerate_devices();
    for (const auto& d : devices) {
        EXPECT_GE(d.max_channels, 0) << d.name;
    }
}

TEST(AudioDevice, SampleRateInRange) {
    auto devices = AudioCapture::enumerate_devices();
    for (const auto& d : devices) {
        EXPECT_GE(d.sample_rate, 8000.0) << d.name << " rate too low";
        EXPECT_LE(d.sample_rate, 384000.0) << d.name << " rate too high";
    }
}

TEST(AudioDevice, ToStringNotEmpty) {
    auto devices = AudioCapture::enumerate_devices();
    for (const auto& d : devices) {
        EXPECT_FALSE(d.to_string().empty());
    }
}

TEST(AudioDevice, ToStringContainsDeviceName) {
    auto devices = AudioCapture::enumerate_devices();
    for (const auto& d : devices) {
        std::string s = d.to_string();
        EXPECT_NE(s.find(d.name), std::string::npos)
            << "to_string() should contain the device name";
    }
}
