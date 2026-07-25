#include <wx/wx.h>
#include "app_frame.h"
#include "device_detect.h"

class WhisperApp : public wxApp {
public:
    virtual bool OnInit() override {
        wxString model_path;
        int device_index = kDeviceAuto;
        bool user_set_device = false;

        for (int i = 1; i < argc; ++i) {
            wxString arg(argv[i]);
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--device" && i + 1 < argc) {
                long val;
                if (wxString(argv[++i]).ToLong(&val)) {
                    device_index = static_cast<int>(val);
                    user_set_device = true;
                }
            } else if (arg == "--cpu") {
                device_index = kDeviceCPU;
                user_set_device = true;
            } else if (arg == "--list-devices") {
                auto devices = whisper_xpu::get_available_devices();
                wxPuts("Available devices:");
                for (const auto& d : devices) {
                    wxPuts(wxString::Format("  [%d] %s", d.index, d.to_string()));
                }
                return false;
            } else if (arg == "--help") {
                wxPuts("Usage: whisper_xpu [options]");
                wxPuts("  --model <path>        Path to model file");
                wxPuts("  --device <index>      Device index (-1=CPU, 0+=GPU)");
                wxPuts("  --cpu                 Shortcut for --device -1");
                wxPuts("  --list-devices        List available devices and exit");
                return false;
            }
        }

        if (!user_set_device) {
            auto devices = whisper_xpu::get_available_devices();
            device_index = kDeviceCPU;
            for (const auto& d : devices) {
                if (d.device_class == DeviceClass::GPU_Discrete) {
                    device_index = d.index;
                    break;
                }
            }
            if (device_index < 0) {
                for (const auto& d : devices) {
                    if (d.device_class == DeviceClass::GPU_Integrated) {
                        device_index = d.index;
                        break;
                    }
                }
            }
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
