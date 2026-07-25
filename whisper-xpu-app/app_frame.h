#pragma once

#include <wx/wx.h>
#include <wx/textctrl.h>
#include <wx/tglbtn.h>
#include <wx/choice.h>
#include <wx/stattext.h>
#include <wx/statusbr.h>
#include <memory>
#include <atomic>
#include <thread>

namespace whisper_xpu {
    class Engine;
    struct DeviceInfo;
}

class AudioCapture;

class AppFrame : public wxFrame {
public:
    AppFrame(const wxString& title, const wxPoint& pos, const wxSize& size,
             const std::string& model_path, int device_index);
    virtual ~AppFrame();

private:
    void OnToggleRecord(wxCommandEvent& event);
    void OnSelectModel(wxCommandEvent& event);
    void OnBrowseModel(wxCommandEvent& event);
    void OnSelectDevice(wxCommandEvent& event);
    void OnSelectMic(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);

    void CreateControls();
    void PopulateModelList();
    void PopulateDeviceList();
    void PopulateMicList();
    void LogMessage(const wxString& text);
    void SetRecording(bool active);
    bool LoadEngine(const std::string& model_path);

    wxChoice*       m_modelChoice;
    wxChoice*       m_deviceChoice;
    wxChoice*       m_micChoice;
    wxToggleButton* m_recordBtn;
    wxButton*       m_browseBtn;
    wxTextCtrl*     m_outputText;
    wxTextCtrl*     m_logText;
    wxStatusBar*    m_statusBar;
    wxString        m_modelDir;

    std::unique_ptr<whisper_xpu::Engine> m_engine;
    std::unique_ptr<AudioCapture> m_audioCapture;
    std::string m_modelPath;
    int m_deviceIndex;
    int m_micIndex = -1;   // -1 = system default
    std::atomic<bool> m_recording{false};
    std::thread m_audioThread;

    enum {
        ID_TRANSCRIBE_RESULT = wxID_HIGHEST + 1,
        ID_TRANSCRIBE_ERROR,
        ID_DEVICE_CHOICE,
        ID_MIC_CHOICE,
    };

    wxDECLARE_EVENT_TABLE();
};
