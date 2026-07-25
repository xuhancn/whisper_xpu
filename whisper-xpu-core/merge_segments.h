#include "export.h"
#pragma once

#include <string>
#include <vector>

namespace whisper_xpu {

// Merge transcribed text segments, deduplicating overlapping suffixes.
// Returns a single string with duplicates removed.
WHISPER_XPU_API std::string merge_segments(const std::vector<const char*>& segments);
WHISPER_XPU_API std::string merge_segments(const std::vector<std::string>& segments);

} // namespace whisper_xpu
