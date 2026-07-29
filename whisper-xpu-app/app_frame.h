#pragma once

#include <wx/wx.h>
#include <wx/textctrl.h>
#include <wx/statusbr.h>
#include <wx/clipbrd.h>
#include <memory>
#include <atomic>
#include "device_detect.h"
#include "audio_capture.h"

namespace whisper_xpu {
    class Engine;
    struct DeviceInfo;
}

class AudioCapture;
class TranscriptionScheduler;

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
    wxButton*       m_recordBtn;        // start / stop recording
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
    // Disable the Record button + show a busy/loading state in the status bar
    // while the engine is being (re)loaded (Engine ctor + GPU warmup can take
    // ~14s on first-kernel JIT).  Keeps the user from starting a recording
    // into a half-loaded engine, and signals "not ready" instead of a frozen
    // window.  Record stays disabled until the engine is ready.
    void SetLoading(bool loading);
    // One-time SYCL/GPU warmup: runs a tiny throwaway whisper_full on the main
    // thread so the Level Zero runtime resolves the decode kernels BEFORE the
    // scheduler's 4 worker threads issue their first GPU compute.  Without it,
    // the workers' first concurrent GPU decode fails ("whisper_full_with_state:
    // failed to decode") and the app AVs in sycl8.dll (0xc0000005).  Verified
    // via tests/streaming_pipeline/test_app_seq{,_warm}.cpp: no-warmup → 0 chars
    // + 84s hang/AV; warmup → real text, ~206ms/window.  test_pipeline passes
    // only because it calls transcribe_file() before start_no_capture().
    void WarmupGpu();

    // ── Engine / audio ──
    std::unique_ptr<whisper_xpu::Engine> m_engine;
    std::unique_ptr<TranscriptionScheduler> m_scheduler;
    std::atomic<bool> m_recording{false};
    std::atomic<bool> m_gpuWarmed{false};
    std::atomic<bool> m_loading{false};   // engine (re)loading + warmup in progress

    wxDECLARE_EVENT_TABLE();
};
