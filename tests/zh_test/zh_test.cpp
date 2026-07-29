// Standalone test for the Chinese-display converter (zh_converter).
// Compiles zh_converter.cpp + zh_t2s.cpp directly (no DLL) and checks that
// Traditional→Simplified and Simplified→Traditional conversions work on
// known text.  Run: zh_test [data-dir]
#include "zh_converter.h"
#include "zh_t2s.h"

#include <cstdio>
#include <string>

static int fails = 0;
static void check(const char* name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++fails;
}
static bool contains(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

int main(int argc, char**) {
    ZhConverter& c = zh_converter();

    // Traditional sample containing 繁體 / 軟體 / 電腦 / 會 / 國.
    const std::string trad = "\xe7\xb9\x81\xe9\xab\x94\xe8\xbb\x9f\xe9\xab\x94"  // 繁體軟體
                            "\xe9\x9b\xbb\xe8\x85\xa6"                            // 電腦
                            "\xe6\x9c\x83\xe8\xad\xb0";                           // 會議
    // Expected simplified glyphs: 繁体软件 电脑 会议
    const char* simp_g = "\xe7\xb9\x81\xe4\xbd\x93\xe8\xbd\xaf\xe4\xbb\xb6";  // 繁体软件

    c.set_mode(ZhConverter::Mode::Auto);
    auto auto_out = c.convert(trad);
    check("Auto is pass-through (untouched)", auto_out == trad);

    c.set_mode(ZhConverter::Mode::Simplified);
    auto simp = c.convert(trad);
    std::printf("  Simplified out: %s\n", simp.c_str());
    check("T→S contains 繁体软件", contains(simp, "\xe7\xb9\x81\xe4\xbd\x93"));  // 繁体
    check("T→S contains 电脑",     contains(simp, "\xe7\x94\xb5\xe8\x84\x91")); // 电脑
    check("T→S contains 会议",     contains(simp, "\xe4\xbc\x9a\xe8\xae\xae")); // 会议

    // Simplified → Traditional
    c.set_mode(ZhConverter::Mode::Traditional);
    auto trad2 = c.convert(simp);
    std::printf("  Traditional out: %s\n", trad2.c_str());
    check("S→T contains 繁體", contains(trad2, "\xe7\xb9\x81\xe9\xab\x94")); // 繁體
    check("S→T contains 軟體", contains(trad2, "\xe8\xbb\x9f\xe9\xab\x94")); // 軟體
    check("S→T contains 電腦", contains(trad2, "\xe9\x9b\xbb\xe8\x85\xa6"));  // 電腦

    // English untouched (CJK gate).
    c.set_mode(ZhConverter::Mode::Simplified);
    auto en = c.convert("Hello world, this is English.");
    check("English untouched by T→S", en == "Hello world, this is English.");

    std::printf("\n=== %s (fails=%d) ===\n", fails ? "SOME FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
