#pragma once

#include <string>

// Pluggable log sink so the scheduler can compile into BOTH:
//   - the wx app  (sink wired to wxLogMessage in main.cpp), and
//   - a headless console unit test (default sink → fprintf(stderr)).
// Keeps <whisper-xpu-app/src/transcription_scheduler.cpp> free of <wx/log.h>.

using SchedLogSink = void(*)(const std::string&);

// Install the sink (call once at startup).  nullptr ⇒ default stderr sink.
void sched_set_log_sink(SchedLogSink sink);

// printf-style; formats locally then hands the whole line to the sink.
// (sic) safe to call before set_sink — goes to stderr.
void sched_log(const char* fmt, ...);
