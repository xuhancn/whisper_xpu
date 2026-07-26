#pragma once

#include <wx/wx.h>
#include <wx/textctrl.h>
#include <wx/tglbtn.h>
#include <wx/statusbr.h>
#include <wx/clipbrd.h>
#include <memory>
#include <atomic>
#include <thread>
#include "device_detect.h"
#include "audio_capture.h"

namespace whisper_xpu {
    class Engine;
    struct DeviceInfo;
}

class AudioCapture;

// ── Status bar field indices ──
// Layout: [Mic info       ] [Device     ] [Model           ] [⚙]
enum StatusField {
    STATUS_MIC      = 0,
    STATUS_DEVICE   = 1,
    STATUS_MODEL    = 2,
    STATUS_SETTINGS = 3,
    STATUS_FIELDS_COUNT
};

class AppFrame : public wxFrame {
public:
    AppFrame(const wxString& title, const wxPoint& pos, const wxSize& size,
             const std::string& model_path, int device_index);
    virtual ~AppFrame();

private:
    void CreateControls();
    void CreateStatusBarFields();
    void ShowSettingsDialog();

    // Event handlers
    void OnToggleRecord(wxCommandEvent& event);
    void OnClear(wxCommandEvent& event);
    void OnCopy(wxCommandEvent& event);
    void OnStatusBarClick(wxMouseEvent& event);
    void OnIdleInit(wxIdleEvent& event);
    void OnClose(wxCloseEvent& event);

    // ── UI controls ──
    wxTextCtrl*     m_transcriptText;   // main editable transcription area
    wxToggleButton* m_recordBtn;        // start / stop recording
    wxButton*       m_clearBtn;         // clear transcription text
    wxButton*       m_copyBtn;          // copy to clipboard

    // ── Settings state ──
    int         m_micIndex    = kMicDefault;
    int         m_deviceIndex = kDeviceAuto;
    std::string m_modelPath;
    wxString    m_hotkeyStr   = "Ctrl+Shift+R";

    // ── Cached lists (populated once at startup) ──
    std::vector<whisper_xpu::DeviceInfo> m_deviceList;
    std::vector<AudioDeviceInfo>         m_micList;

    // ── Helpers ──
    void UpdateStatusBar();
    bool LoadEngine(const std::string& path);

    // ── Stubs for future engine / audio ──
    std::unique_ptr<whisper_xpu::Engine> m_engine;
    std::unique_ptr<AudioCapture>        m_audioCapture;
    std::atomic<bool> m_recording{false};
    std::thread       m_audioThread;

    wxDECLARE_EVENT_TABLE();
};
