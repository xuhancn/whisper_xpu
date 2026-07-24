#include <wx/wx.h>
#include "app_frame.h"

class WhisperApp : public wxApp {
public:
    virtual bool OnInit() override {
        wxString model_path;
        bool use_gpu = true;

        // Parse command-line args
        for (int i = 1; i < argc; ++i) {
            wxString arg(argv[i]);
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--cpu") {
                use_gpu = false;
            } else if (arg == "--help") {
                wxPuts("Usage: whisper_xpu [--model <path>] [--cpu]");
                return false;
            }
        }

        AppFrame* frame = new AppFrame(
            "Whisper XPU",
            wxDefaultPosition,
            wxSize(800, 600),
            model_path.ToStdString(),
            use_gpu
        );
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(WhisperApp);
