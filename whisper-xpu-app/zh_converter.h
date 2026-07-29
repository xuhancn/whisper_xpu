// Chinese-display converter: post-processes transcript text to a chosen glyph
// form (Simplified / Traditional / Auto=internal table) before showing it in
// the UI.
//
// whisper.cpp emits a single `zh` language and lets the model pick the glyph
// set per utterance — turbo/large-v3 sometimes produce Traditional glyphs.
// This converter rewrites transcript text to the user's preferred form.
//
// Modes:
//   Auto        — no-op pass-through (model output untouched).  (The built-in
//                 ~4100-char Traditional→Simplified table is available via
//                 the Simplified mode, which uses OpenCC's TSCharacters.txt.)
//   Simplified  — Traditional → Simplified (OpenCC TSPhrases + TSCharacters).
//   Traditional — Simplified → Traditional (OpenCC STPhrases + STCharacters).
//
// Dictionaries are the vendored OpenCC text dicts (third_party/opencc-data),
// loaded lazily on first use.  Conversion uses longest-prefix (greedy)
// matching over UTF-8 graphemes — not a full segmenter, but correct for the
// common cases and matches OpenCC's max-match for non-ambiguous text.  Only
// runs when the text contains CJK ideographs (zh_has_cjk), so English/other
// transcripts pass through untouched.
#pragma once

#include <string>

class ZhConverter {
public:
    enum class Mode { Auto, Simplified, Traditional };

    // Set the active mode.  Dictionaries load lazily on first convert() call
    // for the relevant direction.  Thread-safe: the merger thread calls
    // convert(); mode is set from the GUI thread before recording starts.
    void set_mode(Mode m) { mode_ = m; }
    Mode mode() const { return mode_; }

    // Convert `text` per the active mode.  Returns a new UTF-8 string; on any
    // load/parse error the original text is returned unchanged (best-effort,
    // never throws — this runs on the merger thread).
    std::string convert(const std::string& text);

private:
    Mode mode_ = Mode::Auto;
    // Lazy-loaded dicts (indices into a hash map keyed by UTF-8 string).
    bool t2s_loaded_ = false;
    bool s2t_loaded_ = false;
    void ensure_t2s();
    void ensure_s2t();
};

// Global instance used by the app (set from the Settings dialog).  The on_text
// callback calls ZhConverter::convert() on each emitted chunk.
ZhConverter& zh_converter();
