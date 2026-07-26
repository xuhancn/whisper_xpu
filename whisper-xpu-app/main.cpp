#include <wx/wx.h>
#include <wx/intl.h>
#include "app_frame.h"
// included via whisper_xpu_core.h

class WhisperApp : public wxApp {
    wxLocale m_locale;
public:
    virtual bool OnInit() override {
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

        // Note: SYCL runtime init (get_available_devices) is deferred to avoid
        // access-violation crashes when the SYCL runtime isn't fully set up
        // (e.g. missing Intel GPU driver or Level Zero loader). The settings
        // dialog will enumerate devices when opened.

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
