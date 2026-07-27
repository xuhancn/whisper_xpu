#include "sched_log.h"

#include <cstdarg>
#include <cstdio>

static SchedLogSink g_sink = nullptr;

void sched_set_log_sink(SchedLogSink sink) { g_sink = sink; }

void sched_log(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    // Clamp to what fit (vsnprintf may have wanted more than the buffer).
    std::string s(buf, (n < (int)sizeof(buf)) ? (size_t)n : sizeof(buf) - 1);
    if (g_sink) g_sink(s);
    else std::fprintf(stderr, "%s\n", s.c_str());
}
