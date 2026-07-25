#pragma once

#include <string>
#include <vector>

// Returns a list of human-readable device descriptions for all
// available compute devices (GPUs via SYCL, plus CPU fallback).
std::vector<std::string> detect_devices();

// Returns true if at least one Intel GPU is available via SYCL
bool has_intel_gpu();

// Returns a short string identifying the best compute device
// (prefers Intel GPU, falls back to CPU)
std::string get_best_device_name();
