#include <wx/wx.h>
#include <wx/intl.h>
#include <wx/evtloop.h>
#include <wx/config.h>
#include <wx/fileconf.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <cstdlib>
#include <cstdio>
#include <io.h>
#include "app_frame.h"
#include "src/sched_log.h"
// included via whisper_xpu_core.h

class WhisperApp : public wxApp {
    wxLocale m_locale;
public:
    virtual bool OnInit() override {
        // Enable wxLog output to stderr for debugging.
        //
        // wxLogStderr writes to stderr, but on a Windows GUI app
        // wxAppTraits::HasStderr() returns false, so DoLogText ALSO re-emits
        // each line via wxMessageOutputDebug — which on MSW writes stderr
        // again.  When stderr is redirected to a file (our usual capture
        // mode: run_app_log.bat uses `2> file`), every line ends up doubled.
        //
        // Fix: hand wxLogStderr a FILE* that is NOT stderr (a dup of stderr's
        // fd, pointing at the same redirected file) so the m_fp==stderr debug
        // re-emit branch is skipped, while output still lands in the same log.
        // _dup fails when stderr isn't redirected (GUI app launched from
        // Explorer) — then the ctor falls back to stderr (logs lost anyway in
        // that mode, no regression).
        FILE* logfp = nullptr;
        int dupfd = _dup(_fileno(stderr));
        if (dupfd != -1) {
            logfp = _fdopen(dupfd, "w");
        }
        wxLog::SetActiveTarget(new wxLogStderr(logfp));

        // Route the TranscriptionScheduler's pluggable log sink (which is
        // wx-free so it can also run in a headless unit test) into wxLog, so
        // its [Scheduler] lines land in the same stderr log as the rest of
        // the app.  The headless test leaves the sink unset (→ stderr).
        sched_set_log_sink([](const std::string& s) {
            wxLogMessage("%s", s.c_str());
        });
        // The SYCL/oneMKL init path inside the merged core library can
        // crash with a null function pointer (sycl::platform::get_platforms)
        // when the Intel GPU driver / Level Zero loader doesn't match the
        // oneAPI runtime version.  We work around this by deferring ALL
        // SYCL calls until after the event loop starts — that way the
        // wxWindow is visible and a crash doesn't look like a hang.

        // Set env var required by Level Zero for sysman function pointers.
#ifdef _WIN32
        _putenv_s("ZES_ENABLE_SYSMAN", "1");
#endif

        m_locale.Init(wxLANGUAGE_DEFAULT);
        wxString model_path;
        int device_index = kDeviceAuto;

        for (int i = 1; i < argc; ++i) {
            wxString arg(argv[i]);
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--device" && i + 1 < argc) {
                long val;
                if (wxString(argv[++i]).ToLong(&val)) {
                    device_index = static_cast<int>(val);
                }
            } else if (arg == "--cpu") {
                device_index = kDeviceCPU;
            } else if (arg == "--help") {
                wxPuts("Usage: whisper_xpu [options]");
                wxPuts("  --model <path>        Path to model file");
                wxPuts("  --device <index>      Device index (-1=CPU, 0+=GPU)");
                wxPuts("  --cpu                 Shortcut for --device -1");
                return false;
            }
        }

        // ZES_ENABLE_SYSMAN=1 is set above — required by the Level Zero
        // loader to avoid a null-function-pointer crash in get_platforms().

        // Set up wxConfigBase (global) backed by whisper_xpu.ini beside the
        // exe, so AppFrame::LoadSettings/SaveSettings persist model/device/
        // mic/zh/hotkey across restarts.  CLI args override the saved values.
        {
            wxFileName ini(wxStandardPaths::Get().GetExecutablePath());
            ini.SetFullName("whisper_xpu.ini");
            wxConfigBase::Set(new wxFileConfig(
                "whisper_xpu", "xuhan",
                ini.GetFullPath(),       // local file (read/write)
                "",                      // no global file
                wxCONFIG_USE_LOCAL_FILE | wxCONFIG_USE_RELATIVE_PATH));
            // Don't auto-create-empty on read; the file appears on first save.
        }

        AppFrame* frame = new AppFrame(
            "Whisper XPU",
            wxDefaultPosition,
            wxSize(800, 600),
            model_path.ToStdString(),
            device_index
        );
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(WhisperApp);
