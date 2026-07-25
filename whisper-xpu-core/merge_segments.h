#pragma once

#include <string>
#include <vector>

namespace whisper_xpu {

// Merge transcribed text segments, deduplicating overlapping suffixes.
// Returns a single string with duplicates removed.
std::string merge_segments(const std::vector<const char*>& segments);
std::string merge_segments(const std::vector<std::string>& segments);

} // namespace whisper_xpu
