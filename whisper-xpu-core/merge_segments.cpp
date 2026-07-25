#include "merge_segments.h"

namespace whisper_xpu {

WHISPER_XPU_API std::string merge_segments(const std::vector<const char*>& segments) {
    std::string result;
    for (size_t i = 0; i < segments.size(); i++) {
        if (!segments[i]) continue;
        std::string cur = segments[i];
        if (cur.empty()) continue;

        if (i > 0 && !result.empty()) {
            size_t rlen = result.size(), clen = cur.size();

            if (clen <= rlen && result.compare(rlen - clen, clen, cur) == 0)
                continue;

            for (size_t o = std::min(rlen, clen); o > 4; o--) {
                if (result.compare(rlen - o, o, cur.c_str(), o) == 0) {
                    cur = cur.substr(o);
                    break;
                }
            }
            if (cur.empty()) continue;
        }
        if (!result.empty() && result.back() != ' ') result += ' ';
        result += cur;
    }
    return result;
}

WHISPER_XPU_API std::string merge_segments(const std::vector<std::string>& segments) {
    std::vector<const char*> ptrs;
    ptrs.reserve(segments.size());
    for (const auto& s : segments) ptrs.push_back(s.c_str());
    return merge_segments(ptrs);
}

} // namespace whisper_xpu
