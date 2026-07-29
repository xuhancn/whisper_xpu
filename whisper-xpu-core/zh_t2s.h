// UTF-8 helpers for Chinese-text gating.  Used by the app's Chinese-display
// converter (whisper-xpu-app/zh_converter) to decide whether a transcript
// chunk is worth running through the Traditional/Simplified converter.
//
// whisper.cpp emits a single `zh` language; the glyph set (Trad/Simp) is the
// model's choice per utterance.  The app post-processes transcript text to a
// chosen display form, but only when the text actually contains CJK ideographs
// so English/other transcripts are never passed through the converter.
#pragma once

#include "export.h"

#include <string>

namespace whisper_xpu {

// True if `utf8` contains at least one CJK Unified Ideograph (U+4E00–U+9FFF).
WHISPER_XPU_API bool zh_has_cjk(const std::string& utf8);

// Decode one UTF-8 codepoint starting at index `i` of `utf8`.  On return `i`
// is advanced past the consumed bytes.  Malformed input yields 0xFFFD and
// advances by 1 (best-effort, no throw).
WHISPER_XPU_API unsigned decode_utf8(const std::string& utf8, size_t& i);

// Append codepoint `cp` as UTF-8 to `out` (handles BMP + supplementary).
WHISPER_XPU_API void append_utf8(std::string& out, unsigned cp);

} // namespace whisper_xpu
