#pragma once

#include <wx/wx.h>
#include <wx/textctrl.h>
#include <wx/statusbr.h>
#include <wx/clipbrd.h>
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include "device_detect.h"
#include "audio_capture.h"
#include "src/transcription_scheduler.h"   // SchedulerStatus, SchedulerState

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

// AppFrame is the VIEW layer in the Model-View-Sync split.  The scheduler is
// a pure data/command API; AppFrame owns a status-sync thread that polls
// query_status() every 100ms and marshals changes to the UI via CallAfter.
// The wx main thread is a pure event loop — it renders SchedulerStatus and
// sends start/stop/reload commands; it never touches scheduler internals.
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
    wxButton*       m_recordBtn;        // start / stop recording
    wxButton*       m_clearBtn;         // clear transcription text
    wxButton*       m_copyBtn;          // copy to clipboard

    // ── Settings state (persisted to whisper_xpu.ini) ──
    int         m_micIndex    = kMicDefault;
    int         m_deviceIndex = kDeviceAuto;
    std::string m_modelPath;
    wxString    m_hotkeyStr   = "Ctrl+Shift+R";

    // ── Cached lists (populated once at startup) ──
    std::vector<whisper_xpu::DeviceInfo> m_deviceList;
    std::vector<AudioDeviceInfo>         m_micList;

    // ── Model-View-Sync ──
    // The scheduler: pure data/command API.  AppFrame owns it but only calls
    // its public methods (start/stop/reload/query_status) from the wx thread
    // and the sync thread.
    std::unique_ptr<TranscriptionScheduler> m_scheduler;
    std::atomic<bool> m_recording{false};

    // Status-sync thread (owned by the UI, NOT the scheduler).  Polls
    // query_status() @100ms; on change, marshals the snapshot to RefreshUI
    // via CallAfter.  Stopped in OnClose before the scheduler is destroyed.
    std::thread      m_syncThread;
    std::atomic<bool> m_stopSync{false};
    SchedulerStatus   m_lastStatus;   // for no-op-poll skip (compared w/ operator==)
    void sync_loop();
    void RefreshUI(const SchedulerStatus& s);   // runs on the wx thread (CallAfter)

    // Render the status bar (mic + device + model).  Called by RefreshUI and
    // by the static init paths.
    void UpdateStatusBar();
    // Load/save the persistent settings (model/device/mic/zh/hotkey) to
    // whisper_xpu.ini so selection survives app restarts.  The reload itself
    // is immediate (Settings OK → m_scheduler->reload); the file is just for
    // the next launch.
    void LoadSettings();
    void SaveSettings() const;

    wxDECLARE_EVENT_TABLE();
};
