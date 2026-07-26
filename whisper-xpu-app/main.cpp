#include <wx/wx.h>
#include <wx/intl.h>
#include <wx/evtloop.h>
#include <cstdlib>
#include "app_frame.h"
// included via whisper_xpu_core.h

class WhisperApp : public wxApp {
    wxLocale m_locale;
public:
    virtual bool OnInit() override {
        // Enable wxLog output to stderr for debugging
        wxLog::SetActiveTarget(new wxLogStderr());
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
