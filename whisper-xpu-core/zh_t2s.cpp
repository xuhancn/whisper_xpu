// UTF-8 helpers for Chinese-text gating.  See zh_t2s.h.
#include "zh_t2s.h"

#include <cstdint>

namespace whisper_xpu {

unsigned decode_utf8(const std::string& utf8, size_t& i) {
    const unsigned char* s = reinterpret_cast<const unsigned char*>(utf8.data());
    size_t n = utf8.size();
    if (i >= n) { i = n; return 0xFFFD; }
    unsigned char c = s[i];
    if (c < 0x80) { i += 1; return c; }
    if ((c & 0xE0) == 0xC0) {
        if (i + 1 < n) {
            unsigned cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
            i += 2; return cp;
        }
        i += 1; return 0xFFFD;
    }
    if ((c & 0xF0) == 0xE0) {
        if (i + 2 < n) {
            unsigned cp = ((c & 0x0F) << 12)
                        | ((s[i+1] & 0x3F) << 6)
                        |  (s[i+2] & 0x3F);
            i += 3; return cp;
        }
        i += 1; return 0xFFFD;
    }
    if ((c & 0xF8) == 0xF0) {
        if (i + 3 < n) {
            unsigned cp = ((c & 0x07) << 18)
                        | ((s[i+1] & 0x3F) << 12)
                        | ((s[i+2] & 0x3F) << 6)
                        |  (s[i+3] & 0x3F);
            i += 4; return cp;
        }
        i += 1; return 0xFFFD;
    }
    i += 1; return 0xFFFD;
}

void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

bool zh_has_cjk(const std::string& utf8) {
    size_t i = 0;
    while (i < utf8.size()) {
        unsigned cp = decode_utf8(utf8, i);
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    }
    return false;
}

} // namespace whisper_xpu
