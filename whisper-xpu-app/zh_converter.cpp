// Chinese-display converter.  See zh_converter.h.
//
// Reads the vendored OpenCC text dictionaries (third_party/opencc-data) at
// runtime and does longest-prefix matching.  No external library dependency.

#include "zh_converter.h"
#include "zh_t2s.h"            // whisper_xpu: decode_utf8/append_utf8/zh_has_cjk

#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

// Vendored OpenCC data, resolved to an absolute path by CMake at configure
// time (WHISPER_XPU_OPENCC_DATA_DIR).  Defaults to a relative path for
// out-of-CMake dev runs.
#ifdef WHISPER_XPU_OPENCC_DATA_DIR
static const std::string g_data_dir = WHISPER_XPU_OPENCC_DATA_DIR;
#else
static const std::string g_data_dir = "third_party/opencc-data";
#endif

// One direction's dicts: phrases (long) + characters (single grapheme).
// Looked up by UTF-8 key → replacement (first value).
struct DictSet {
    bool loaded = false;
    // ordered-ish: phrases can be multi-grapheme; characters single.
    std::unordered_map<std::string, std::string> phrases;
    std::unordered_map<std::string, std::string> chars;
    size_t max_key_len = 0;   // in UTF-8 bytes, for prefix-window sizing
};

DictSet g_t2s;   // Traditional → Simplified
DictSet g_s2t;   // Simplified → Traditional
std::mutex g_mu;

// Read one OpenCC .txt dict: lines "key\tvalue1 value2 ...".  Comment lines
// start with '#'.  Blank lines ignored.  The first value is taken (OpenCC's
// short_circuit policy uses the first candidate).
bool load_dict(const std::string& rel, std::unordered_map<std::string,std::string>& out,
               size_t& max_key_len, bool is_phrase) {
    std::string path = g_data_dir + "/dictionary/" + rel;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "[zh] cannot open %s\n", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        // strip trailing \r (CRLF)
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        // key \t value1 [ value2 ...]
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string key = line.substr(0, tab);
        std::string rest = line.substr(tab + 1);
        // first value (space-separated)
        size_t sp = rest.find(' ');
        std::string val = (sp == std::string::npos) ? rest : rest.substr(0, sp);
        if (key.empty() || val.empty()) continue;
        // Only keep the first mapping per key (short_circuit).
        const size_t klen = key.size();   // capture before std::move(key)
        out.emplace(std::move(key), std::move(val));
        if (klen > max_key_len) max_key_len = klen;
    }
    return true;
}

void load_t2s(DictSet& d) {
    // Order doesn't matter for lookup (we try longest prefix), but phrases
    // generally take priority via max-prefix matching.
    load_dict("TSPhrases.txt", d.phrases, d.max_key_len, true);
    load_dict("TSCharacters.txt", d.chars, d.max_key_len, false);
    d.loaded = true;
}

void load_s2t(DictSet& d) {
    load_dict("STPhrases.txt", d.phrases, d.max_key_len, true);
    load_dict("STCharacters.txt", d.chars, d.max_key_len, false);
    d.loaded = true;
}

// Grapheme length in bytes of one UTF-8 char starting at s[i].
size_t grapheme_bytes(const std::string& s, size_t i) {
    if (i >= s.size()) return 0;
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return (i + 1 < s.size()) ? 2 : 1;
    if ((c & 0xF0) == 0xE0) return (i + 2 < s.size()) ? 3 : (s.size() - i);
    if ((c & 0xF8) == 0xF0) return (i + 3 < s.size()) ? 4 : (s.size() - i);
    return 1;
}

// Convert using a loaded DictSet: longest-prefix match (greedy).  Walks the
// input UTF-8; at each position tries the longest dictionary key that fits,
// preferring phrases over single chars (handled by trying max_key_len down).
std::string convert_with(const std::string& text, const DictSet& d) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        // Longest-prefix match: try the longest grapheme-aligned substring
        // (capped at max_key_len bytes) that is a dict key.  Phrases and
        // single chars are both checked; longest wins.
        size_t cap = std::min(text.size() - i, d.max_key_len);
        size_t best_len = 0;
        std::string best_val;
        size_t len = 0;
        while (len < cap) {
            size_t at = i + len;
            if (at >= text.size()) break;
            size_t gb = grapheme_bytes(text, at);
            if (gb == 0) break;          // malformed / EOL guard
            len += gb;
            if (len > cap) break;        // window exceeded max_key_len
            std::string key = text.substr(i, len);
            auto pit = d.phrases.find(key);
            if (pit != d.phrases.end()) {
                best_len = len;
                best_val = pit->second;
                continue;               // keep scanning for a longer phrase
            }
            auto cit = d.chars.find(key);
            if (cit != d.chars.end()) {
                best_len = len;
                best_val = cit->second;
            }
        }
        if (best_len > 0) {
            out += best_val;
            i += best_len;
        } else {
            // No dict match: copy one grapheme verbatim.
            size_t gb = grapheme_bytes(text, i);
            if (gb == 0) gb = 1;
            out.append(text, i, gb);
            i += gb;
        }
    }
    return out;
}

} // namespace

ZhConverter& zh_converter() {
    static ZhConverter inst;
    return inst;
}

std::string ZhConverter::convert(const std::string& text) {
    if (mode_ == Mode::Auto) return text;   // Auto: untouched (model output)
    // Only run on text that actually contains CJK (cheap gate; English etc.
    // passes through).  zh_has_cjk is exported by whisper_xpu_sycl_core.
    if (!whisper_xpu::zh_has_cjk(text)) return text;

    std::lock_guard<std::mutex> lk(g_mu);
    if (mode_ == Mode::Simplified) {
        if (!g_t2s.loaded) load_t2s(g_t2s);
        if (!g_t2s.loaded) return text;          // load failed → best-effort
        return convert_with(text, g_t2s);
    } else { // Traditional
        if (!g_s2t.loaded) load_s2t(g_s2t);
        if (!g_s2t.loaded) return text;
        return convert_with(text, g_s2t);
    }
}
