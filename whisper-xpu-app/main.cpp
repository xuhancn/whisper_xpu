#include <wx/wx.h>
#include <wx/intl.h>
#include <cstdlib>
#include "app_frame.h"
// included via whisper_xpu_core.h

class WhisperApp : public wxApp {
    wxLocale m_locale;
public:
    virtual bool OnInit() override {
        // Level Zero sysman must be enabled before any SYCL call or the
        // runtime hits a null function pointer in ze_loader.dll during
        // platform enumeration.
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
