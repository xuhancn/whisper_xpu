#include "app_frame.h"
#include "whisper_xpu_core.h"
#include "device_detect.h"
#include "audio_capture.h"
#include "src/transcription_scheduler.h"
#include "zh_converter.h"

#include <wx/filedlg.h>
#include <wx/msgdlg.h>
#include <wx/statbox.h>
#include <wx/filename.h>
#include <wx/icon.h>
#include <wx/image.h>
#include <wx/stdpaths.h>
#include <wx/config.h>      // wxFileConfig-backed wxConfigBase for whisper_xpu.ini
#include <wx/log.h>
#include <chrono>
#include <thread>

wxBEGIN_EVENT_TABLE(AppFrame, wxFrame)
    EVT_CLOSE(AppFrame::OnClose)
wxEND_EVENT_TABLE()

// ──────────────────────────────────────────
//  SEH-safe wrappers (no C++ object unwinding)
// ──────────────────────────────────────────

// ── Thin wrapper ──
// get_available_devices() is compiled with icpx inside
// whisper_xpu_sycl_core.dll.  SEH guards inside the DLL catch any
// AV from sycl8.dll — the MSVC caller never sees a crash.
static std::vector<whisper_xpu::DeviceInfo> safe_get_devices() {
    return whisper_xpu::get_available_devices();
}

static std::vector<AudioDeviceInfo> safe_enum_audio() {
    try {
        return AudioCapture::enumerate_devices();
    } catch (const std::exception&) {
        return {};
    }
}

// ──────────────────────────────────────────
//  Construction / Destruction
// ──────────────────────────────────────────

AppFrame::AppFrame(const wxString& title, const wxPoint& pos, const wxSize& size,
                   const std::string& model_path, int device_index)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
    , m_modelPath(model_path)
    , m_deviceIndex(device_index)
{
    CreateControls();
    SetMinSize(wxSize(600, 350));
    SetSize(900, 600);

    // App icon (title bar + taskbar).  Load the 256px PNG co-located beside
    // the exe (CMake POST_BUILD copies resources/*.png next to the exe).  If
    // the file is absent (e.g. running from a build dir without the copy
    // step), silently fall back to the default wx icon.
    {
        wxFileName iconPath(wxStandardPaths::Get().GetExecutablePath());
        iconPath.SetFullName("app256.png");
        if (iconPath.FileExists()) {
            wxInitAllImageHandlers();
            wxIcon icn;
            icn.CopyFromBitmap(wxBitmap(iconPath.GetFullPath(), wxBITMAP_TYPE_PNG));
            SetIcon(icn);
        }
    }

    // Persisted settings: if no --model/--device on the CLI, load the last
    // selection from whisper_xpu.ini so the user's choice survives restarts.
    // (CLI args, when present, still win.)
    LoadSettings();

    // (No ctor-time SYCL usability probe here: calling get_device_info() /
    // get_available_devices() runs a SYCL backend-init probe, and SYCL calls
    // must be deferred until the event loop is running — doing it in the ctor
    // (pre event-loop) stalls startup and the model never loads.  Device
    // usability is checked instead inside the Settings dialog, where
    // m_deviceList is already populated by OnIdleInit (post event-loop) with
    // the usable flag.  If a persisted/CLI device index is unusable, the user
    // changes it in Settings; the load on an unusable device would crash at
    // SYCL init, but every enumerated device's driver currently inits fine.)

    // Populate cached device/mic lists and update status bar.
    // SYCL device enumeration is deferred to OnIdleInit (after event
    // loop starts) via whisper_xpu_sycl_core.dll.  The DLL is
    // delay-loaded so the app starts regardless of sycl8.dll presence.
    m_micList = safe_enum_audio();
    UpdateStatusBar();

    // Construct the scheduler (Model layer) — it OWNS the engine and loads the
    // model on a BACKGROUND thread from its ctor.  This never blocks the GUI
    // thread.  on_text is the merger→UI data delivery callback (marshaled to
    // the wx thread via CallAfter inside the callback — the scheduler itself
    // has no wx knowledge).  The UI observes state via the sync thread below.
    m_scheduler = std::make_unique<TranscriptionScheduler>(
        [this](const std::string& text) {  // on_text: merger thread
            std::string shown = zh_converter().convert(text);
            m_transcriptText->CallAfter([this, shown]() {
                m_transcriptText->AppendText(wxString::FromUTF8(shown) + wxT("\n"));
                m_transcriptText->Update();
            });
        },
        m_modelPath,
        m_deviceIndex);

    // Status-sync thread (View layer's poller).  Polls query_status() @100ms;
    // on change, marshals the snapshot to RefreshUI via CallAfter.  Owned by
    // the UI (the scheduler has no threads-for-UI).  Stopped in OnClose before
    // the scheduler is destroyed.
    m_syncThread = std::thread(&AppFrame::sync_loop, this);

    // Deferred SYCL device enumeration after event loop is running
    Bind(wxEVT_IDLE, &AppFrame::OnIdleInit, this);
}

void AppFrame::OnIdleInit(wxIdleEvent& event) {
    // Run once: unbind immediately so we only fire once.
    Unbind(wxEVT_IDLE, &AppFrame::OnIdleInit, this);

    // Enumerate SYCL devices through the icpx-compiled DLL.  This is
    // deferred until the event loop is running so the window is visible
    // before any SYCL call.  If sycl8.dll is absent or the driver
    // crashes, safe_get_devices() catches it and returns empty.
    m_deviceList = safe_get_devices();
    UpdateStatusBar();

    event.Skip();
}

// ──────────────────────────────────────────
// Status-sync thread (View layer's poller)
// ──────────────────────────────────────────

void AppFrame::sync_loop() {
    // Polls the scheduler's pure query_status() every 100ms; on any change,
    // marshals the snapshot to the wx thread via CallAfter(RefreshUI).  This is
    // the ONLY place that turns scheduler data into UI updates, and it owns no
    // scheduler internals — just the read-only snapshot.  Exits when m_stopSync
    // is set (OnClose/~AppFrame).
    while (!m_stopSync.load()) {
        if (m_scheduler) {
            SchedulerStatus s = m_scheduler->query_status();
            if (s != m_lastStatus) {
                m_lastStatus = s;
                // Capture by value (small POD-ish struct) so the snapshot is
                // stable across the CallAsync boundary regardless of later polls.
                this->CallAfter([this, s]() { RefreshUI(s); });
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void AppFrame::RefreshUI(const SchedulerStatus& s) {
    // Runs on the wx thread (via CallAfter).  Pure view: render the snapshot +
    // send no commands.  State → status-bar text + button label/enable.
    UpdateStatusBar();   // device/model fields reflect the latest snapshot
    switch (s.state) {
        case SchedulerState::Idle:
            SetStatusText("No model loaded", STATUS_MIC);
            m_recordBtn->SetLabel("Record");
            m_recordBtn->Enable(false);
            break;
        case SchedulerState::Loading:
            // Source is compiled /utf-8 so "…" is UTF-8 bytes; the system
            // locale (GBK 936) would mis-decode the ellipsis → garbled
            // "[loading***".  wxString::FromUTF8 decodes the bytes as UTF-8.
            SetStatusText(wxString::FromUTF8("Loading model…"), STATUS_MIC);
            m_recordBtn->SetLabel("Record");
            // Record stays enabled: pressing it during load starts capture
            // immediately; the pipeline launches when warmup finishes.
            m_recordBtn->Enable(true);
            break;
        case SchedulerState::Ready:
            SetStatusText("GPU Ready", STATUS_MIC);
            m_recordBtn->SetLabel("Record");
            m_recordBtn->Enable(true);
            break;
        case SchedulerState::Recording:
            SetStatusText(wxString::FromUTF8("Recording…"), STATUS_MIC);
            m_recordBtn->SetLabel("Stop");
            m_recordBtn->Enable(true);
            m_recording.store(true);
            break;
        case SchedulerState::Failed:
            SetStatusText("Engine load failed", STATUS_MIC);
            m_recordBtn->SetLabel("Record");
            m_recordBtn->Enable(false);
            // Only show the error dialog once per failure: track via a flag
            // on m_lastStatus (Failed → Failed repeats won't re-show because
            // query_status returns the same snapshot ⇒ no re-call).  The first
            // Failed transition surfaces here.
            wxLogMessage("[app] engine load failed (see scheduler log)");
            break;
    }
}

AppFrame::~AppFrame() {
    // Stop the sync thread FIRST so it doesn't query a dying scheduler.  Then
    // stop any recording; the scheduler's dtor (m_scheduler unique_ptr) aborts
    // + joins the load thread and frees the engine.
    m_stopSync.store(true);
    if (m_syncThread.joinable()) m_syncThread.join();
    if (m_recording && m_scheduler)
        m_scheduler->stop();
}

// ──────────────────────────────────────────
//  Layout
// ──────────────────────────────────────────

void AppFrame::CreateControls() {
    auto* panel = new wxPanel(this, wxID_ANY);
    auto* root  = new wxBoxSizer(wxVERTICAL);

    // ── Transcription text (editable) ──
    m_transcriptText = new wxTextCtrl(
        panel, wxID_ANY, "",
        wxDefaultPosition, wxDefaultSize,
        wxTE_MULTILINE | wxTE_RICH2 | wxTE_WORDWRAP
    );
    m_transcriptText->SetFont(
        wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    m_transcriptText->SetToolTip(
        "Transcription output — editable.\n"
        "New transcription text is inserted at the cursor position.");

    root->Add(m_transcriptText, 1, wxEXPAND | wxALL, 10);

    // ── Button bar ──
    auto* btnSizer = new wxBoxSizer(wxHORIZONTAL);

    // Note: avoid color emoji and BMP symbols — Win32 Button control on some
    // Windows builds/skins doesn't render them reliably. Use plain ASCII text.
    m_clearBtn = new wxButton(panel, wxID_ANY, "Clear");
    m_clearBtn->SetMinSize(wxSize(100, 32));
    m_clearBtn->SetToolTip("Clear all transcription text");
    btnSizer->Add(m_clearBtn, 0, wxRIGHT, 8);

    m_recordBtn = new wxButton(panel, wxID_ANY, "Record");
    m_recordBtn->SetMinSize(wxSize(100, 32));
    m_recordBtn->SetToolTip("Start / stop recording");
    btnSizer->Add(m_recordBtn, 0, wxRIGHT, 8);

    m_copyBtn = new wxButton(panel, wxID_ANY, "Copy");
    m_copyBtn->SetMinSize(wxSize(100, 32));
    m_copyBtn->SetToolTip("Copy transcription to clipboard");
    btnSizer->Add(m_copyBtn, 0);

    root->Add(btnSizer, 0, wxALIGN_CENTER | wxBOTTOM, 10);

    panel->SetSizer(root);

    // ── 4-segment status bar ──
    CreateStatusBarFields();

    // Force frame to re-layout now that the status bar exists — the panel
    // was sized before the status bar was created, so without this the
    // bottom of the panel may be hidden behind the status bar.
    Layout();

    // ── Events ──
    m_clearBtn->Bind(wxEVT_BUTTON, &AppFrame::OnClear, this);
    m_recordBtn->Bind(wxEVT_BUTTON, &AppFrame::OnToggleRecord, this);
    m_copyBtn->Bind(wxEVT_BUTTON, &AppFrame::OnCopy, this);
    GetStatusBar()->Bind(wxEVT_LEFT_DOWN, &AppFrame::OnStatusBarClick, this);

    // Record starts DISABLED — enabled by RefreshUI once the sync thread
    // reports a usable state (Loading/Ready).  RefreshUI (driven by the sync
    // thread's query_status poll) owns all button enable/disable + label.
    m_recordBtn->Disable();
}

void AppFrame::CreateStatusBarFields() {
    CreateStatusBar(STATUS_FIELDS_COUNT);

    // Widths: negative = stretch proportional, positive = fixed px
    int widths[STATUS_FIELDS_COUNT] = { -1, 130, -1, 70 };
    SetStatusWidths(STATUS_FIELDS_COUNT, widths);

    SetStatusText("Mic: Default",  STATUS_MIC);
    SetStatusText("Device: Auto",  STATUS_DEVICE);
    SetStatusText("No model",      STATUS_MODEL);
    SetStatusText("Settings...",   STATUS_SETTINGS);
}

// ──────────────────────────────────────────
//  Status bar helpers
// ──────────────────────────────────────────

void AppFrame::UpdateStatusBar() {
    // All rendered strings are UTF-8 bytes (source compiled /utf-8; device/
    // mic/model names are UTF-8).  wxString(const char*) would decode them
    // using the system locale (GBK 936 on Chinese Windows) and garble any
    // non-ASCII (e.g. the "…" ellipsis, or a localized device/mic name).  Use
    // wxString::FromUTF8 so the bytes are decoded as UTF-8 regardless of the
    // system code page.
    // Mic field — show actual device name, append (Default) if system default
    wxString micLabel = wxT("Mic: N/A");
    if (m_micIndex == kMicDefault) {
        for (const auto& m : m_micList) {
            if (m.is_default) {
                micLabel = wxT("Mic: ") + wxString::FromUTF8(m.name) + wxT(" (Default)");
                break;
            }
        }
        if (micLabel == wxT("Mic: N/A") && !m_micList.empty()) {
            micLabel = wxT("Mic: ") + wxString::FromUTF8(m_micList[0].name);
        }
    } else {
        for (const auto& m : m_micList) {
            if (m.index == m_micIndex) {
                micLabel = wxT("Mic: ") + wxString::FromUTF8(m.name);
                break;
            }
        }
    }
    SetStatusText(micLabel, STATUS_MIC);

    // Device field — shows CPU/GPU info or Auto if nothing enumerated
    wxString devLabel = wxT("Device: Auto");
    if (m_deviceIndex == kDeviceCPU) {
        devLabel = wxT("Device: CPU");
    } else if (m_deviceIndex >= 0) {
        for (const auto& d : m_deviceList) {
            if (d.index == m_deviceIndex) {
                devLabel = wxT("Device: ") + wxString::FromUTF8(d.to_string());
                break;
            }
        }
    } else if (m_deviceIndex == kDeviceAuto) {
        // Auto-select: show first GPU if available
        for (const auto& d : m_deviceList) {
            if (d.index >= 0) {
                devLabel = wxT("Device: ") + wxString::FromUTF8(d.to_string()) + wxT(" [Auto]");
                break;
            }
        }
    }
    SetStatusText(devLabel, STATUS_DEVICE);

    // Model field — render from the latest status snapshot (owned by the sync
    // thread) or a fresh query.  m_lastStatus is written by the sync thread;
    // reading it here (on the wx thread) is a benign plain read of plain
    // data — RefreshUI is the authoritative renderer, this is just for the
    // static-init paints before the sync thread has posted.
    SchedulerStatus s = m_scheduler ? m_scheduler->query_status() : SchedulerStatus{};
    wxString modelLabel = "No model";
    if (!s.model_name.empty()) {
        modelLabel = wxString::FromUTF8(s.model_name);
        switch (s.state) {
            case SchedulerState::Loading: modelLabel += wxString::FromUTF8(" [loading…]"); break;
            case SchedulerState::Ready:
            case SchedulerState::Recording:
                modelLabel += s.device_desc.find("CPU") == std::string::npos
                              ? " [GPU]" : " [CPU]";
                break;
            case SchedulerState::Failed:  modelLabel += " [failed]"; break;
            default: break;
        }
    }
    SetStatusText(modelLabel, STATUS_MODEL);
}

// ──────────────────────────────────────────
//  Persistent settings (whisper_xpu.ini)
// ──────────────────────────────────────────

void AppFrame::LoadSettings() {
    // wxConfigBase (set up in main.cpp's OnInit as a wxFileConfig backed by
    // whisper_xpu.ini beside the exe).  CLI --model/--device override the
    // persisted values (checked in the ctor before calling this): if a CLI
    // model_path was given, keep it; otherwise load the saved one.
    wxConfigBase* cfg = wxConfigBase::Get();
    if (!cfg) return;
    if (m_modelPath.empty())
        m_modelPath = cfg->Read("model", "").ToStdString();
    if (m_deviceIndex == kDeviceAuto) {
        // Only fall back to the saved device if the CLI didn't force one.  The
        // ctor receives kDeviceAuto by default; a CLI --device sets a real one.
        long saved = cfg->ReadLong("device", kDeviceAuto);
        m_deviceIndex = (int)saved;
    }
    m_micIndex  = (int)cfg->ReadLong("mic",   kMicDefault);
    wxString hk = cfg->Read("hotkey", m_hotkeyStr);
    if (!hk.IsEmpty()) m_hotkeyStr = hk;
    long zh = cfg->ReadLong("zh", -1);
    if (zh >= 0 && zh <= 2)
        zh_converter().set_mode(static_cast<ZhConverter::Mode>(zh));
}

void AppFrame::SaveSettings() const {
    wxConfigBase* cfg = wxConfigBase::Get();
    if (!cfg) return;
    cfg->Write("model",  wxString(m_modelPath));
    cfg->Write("device", (long)m_deviceIndex);
    cfg->Write("mic",    (long)m_micIndex);
    cfg->Write("hotkey", m_hotkeyStr);
    cfg->Write("zh",     (long)static_cast<int>(zh_converter().mode()));
    cfg->Flush();
}

// ──────────────────────────────────────────
//  Settings Dialog
// ──────────────────────────────────────────

void AppFrame::ShowSettingsDialog() {
    wxDialog dlg(this, wxID_ANY, "Settings",
                 wxDefaultPosition, wxSize(500, 480),
                 wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);
    auto* root = new wxBoxSizer(wxVERTICAL);

    // ── Microphone ──
    std::vector<wxString> micLabels;
    std::vector<int>       micIndices;
    micLabels.push_back("System Default");
    micIndices.push_back(kMicDefault);

    int micSel = 0;
    for (size_t i = 0; i < m_micList.size(); ++i) {
        wxString name(m_micList[i].name);
        if (name.Trim().IsEmpty()) continue;
        if (m_micList[i].index == m_micIndex) micSel = (int)micLabels.size();
        micLabels.push_back(name);
        micIndices.push_back((int)i);
    }

    auto* micBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Microphone");
    auto* micChoice = new wxChoice(micBox->GetStaticBox(), wxID_ANY);
    for (size_t i = 0; i < micLabels.size(); ++i)
        micChoice->Append(micLabels[i]);
    micChoice->SetSelection(micSel);
    micBox->Add(micChoice, 1, wxEXPAND | wxALL, 4);
    root->Add(micBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

    // ── XPU Device ──
    // Uses wxListBox (not wxChoice) so unusable devices can be GREYED OUT
    // (wxChoice can't disable individual rows on Windows).  An iGPU whose
    // driver can't init a SYCL backend (probe_device_usable_raw in
    // device_detect.cpp) shows here but is disabled — the user sees it exists
    // but can't pick it.  Single-select (no wxLB_MULTIPLE).
    std::vector<wxString> devLabels;
    std::vector<int>       devIndices;
    std::vector<bool>      devUsable;   // parallel: can this device be selected?
    devLabels.push_back("Auto");
    devIndices.push_back(kDeviceAuto);
    devUsable.push_back(true);

    int devSel = 0;
    for (size_t i = 0; i < m_deviceList.size(); ++i) {
        wxString label(m_deviceList[i].to_string());
        if (label.Trim().IsEmpty()) continue;
        bool usable = m_deviceList[i].usable;
        if (!usable) label += "  (driver unavailable)";  // visible even when greyed
        if (m_deviceList[i].index == m_deviceIndex) devSel = (int)devLabels.size();
        devLabels.push_back(label);
        devIndices.push_back((int)i);
        devUsable.push_back(usable);
    }

    auto* devBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Device");
    // wxChoice (dropdown): wxWidgets' standard list/choice controls on MSW
    // canNOT disable individual rows (only the whole control).  So unusable
    // devices are NOT hidden — they appear with a "(driver unavailable)"
    // suffix, and if the user picks one the OK handler refuses to apply it
    // (reverts).  This is the closest stable approximation to "greyed out"
    // without swapping in a heavyweight wxDataView/owner-draw control.
    auto* devChoice = new wxChoice(devBox->GetStaticBox(), wxID_ANY);
    for (size_t i = 0; i < devLabels.size(); ++i)
        devChoice->Append(devLabels[i]);
    // If the persisted selection is unusable, fall back to Auto (row 0) so the
    // dialog opens on a valid selection instead of an unusable one.
    if (devSel >= 0 && devSel < (int)devUsable.size() && !devUsable[devSel])
        devSel = 0;
    devChoice->SetSelection(devSel);
    devBox->Add(devChoice, 1, wxEXPAND | wxALL, 4);
    root->Add(devBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

    // ── Model path + browse ──
    auto* modelBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Model");
    auto* modelText = new wxTextCtrl(modelBox->GetStaticBox(), wxID_ANY, wxString(m_modelPath),
                                     wxDefaultPosition, wxDefaultSize,
                                     wxTE_PROCESS_ENTER);
    modelText->SetToolTip("Path to a Whisper model file (.bin / .ggml / .gguf)");
    modelBox->Add(modelText, 1, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    auto* browseBtn = new wxButton(modelBox->GetStaticBox(), wxID_ANY, "Browse...");
    browseBtn->Bind(wxEVT_BUTTON, [&dlg, modelText](wxCommandEvent&) {
        wxFileDialog fd(&dlg, "Open Whisper Model File", "", "",
                        "Model files (*.bin;*.ggml;*.gguf)|*.bin;*.ggml;*.gguf"
                        "|All files (*.*)|*.*",
                        wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (fd.ShowModal() == wxID_OK)
            modelText->SetValue(fd.GetPath());
    });
    modelBox->Add(browseBtn, 0, wxALL, 4);
    root->Add(modelBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

    // ── Chinese display (glyph form for transcribed Chinese) ──
    // whisper lets the model pick Trad/Simp glyphs per utterance; this rewrites
    // transcript text to the chosen form before display.  Auto = untouched.
    auto* zhBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Chinese display");
    auto* zhChoice = new wxChoice(zhBox->GetStaticBox(), wxID_ANY);
    zhChoice->Append("Auto (model output)");
    zhChoice->Append("Simplified");
    zhChoice->Append("Traditional");
    zhChoice->SetSelection(static_cast<int>(zh_converter().mode()));
    zhBox->Add(zhChoice, 1, wxEXPAND | wxALL, 4);
    root->Add(zhBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

    // ── Record Hotkey ──
    auto* hotkeyBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Record Hotkey");
    auto* hotkeyText = new wxTextCtrl(hotkeyBox->GetStaticBox(), wxID_ANY, m_hotkeyStr,
                                      wxDefaultPosition, wxSize(200, -1),
                                      wxTE_READONLY);
    hotkeyText->SetToolTip("Click this field, then press the desired key combination");
    hotkeyText->Bind(wxEVT_KEY_DOWN, [hotkeyText](wxKeyEvent& ev) {
        wxString combo;
        if (ev.ControlDown()) combo += "Ctrl+";
        if (ev.ShiftDown())   combo += "Shift+";
        if (ev.AltDown())     combo += "Alt+";

        int key = ev.GetKeyCode();
        if (key >= WXK_F1 && key <= WXK_F12) {
            combo += wxString::Format("F%d", key - WXK_F1 + 1);
        } else if (key >= 'A' && key <= 'Z') {
            combo += wxString::Format("%c", (char)key);
        } else if (key == WXK_SPACE) {
            combo += "Space";
        } else {
            return;
        }
        hotkeyText->SetValue(combo);
    });
    hotkeyBox->Add(hotkeyText, 0, wxALL, 4);
    hotkeyBox->Add(new wxStaticText(hotkeyBox->GetStaticBox(), wxID_ANY,
                    "(Click field, press key)"),
                   0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    root->Add(hotkeyBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 8);

    // ── OK / Cancel ──
    auto* btnSizer = new wxBoxSizer(wxHORIZONTAL);
    auto* okBtn     = new wxButton(&dlg, wxID_OK,     "OK");
    auto* cancelBtn = new wxButton(&dlg, wxID_CANCEL, "Cancel");
    okBtn->SetMinSize(wxSize(80, 28));
    cancelBtn->SetMinSize(wxSize(80, 28));
    btnSizer->AddStretchSpacer();
    btnSizer->Add(okBtn,     0, wxRIGHT, 8);
    btnSizer->Add(cancelBtn, 0);
    root->Add(btnSizer, 0, wxEXPAND | wxTOP | wxBOTTOM | wxRIGHT, 8);

    dlg.SetSizer(root);
    root->Fit(&dlg);
    dlg.SetMinSize(wxSize(440, dlg.GetSize().y));
    dlg.Centre();

    if (dlg.ShowModal() == wxID_OK) {
        // ── Read mic selection ──
        int newMicIdx = kMicDefault;
        {
            int sel = micChoice->GetSelection();
            if (sel >= 0 && static_cast<size_t>(sel) < micIndices.size()) {
                int idx = micIndices[sel];
                newMicIdx = (idx == kMicDefault) ? kMicDefault : m_micList[idx].index;
            }
        }

        // ── Read device selection ──
        int newDevIdx = kDeviceAuto;
        {
            int sel = devChoice->GetSelection();
            if (sel >= 0 && static_cast<size_t>(sel) < devIndices.size()) {
                int idx = devIndices[sel];
                newDevIdx = (idx == kDeviceAuto) ? kDeviceAuto : m_deviceList[idx].index;
            }
        }

        std::string newModelPath = modelText->GetValue().ToStdString();
        wxString newHotkey       = hotkeyText->GetValue();
        int zhSel = zhChoice->GetSelection();   // -1 if none

        // Refuse an unusable device selection (the row is marked
        // "(driver unavailable)").  Keep the model/mic/zh changes but don't
        // switch the device to one whose driver can't init a SYCL backend.
        if (newDevIdx >= 0) {
            auto di = whisper_xpu::get_device_info(newDevIdx);
            if (!di.usable) {
                wxMessageBox(
                    wxString::FromUTF8("The selected device '") + di.name +
                    wxString::FromUTF8("' is unusable (its driver rejected "
                        "SYCL backend init). Pick the dGPU or CPU."),
                    "Device Unavailable", wxOK | wxICON_WARNING);
                newDevIdx = m_deviceIndex;   // keep the old, usable device
            }
        }

        bool devChanged   = (newDevIdx != m_deviceIndex);
        bool modelChanged = (newModelPath != m_modelPath);

        m_micIndex    = newMicIdx;
        m_deviceIndex = newDevIdx;
        m_modelPath   = newModelPath;
        m_hotkeyStr   = newHotkey;
        // Apply the Chinese-display mode immediately (takes effect on the next
        // emitted chunk).
        if (zhSel >= 0 && zhSel <= 2)
            zh_converter().set_mode(static_cast<ZhConverter::Mode>(zhSel));

        // Persist ALL settings so the next launch comes back to this selection.
        SaveSettings();
        UpdateStatusBar();

        // Immediate reload (never blocks the GUI): the scheduler stops any
        // recording, joins the (already-finished) load thread, drops the old
        // engine, and starts a fresh background load.  The sync thread observes
        // Loading → Ready/Failed and RefreshUI flips the status bar.
        //
        // CRASH GUARD: never reload while a load is in progress.  Aborting GPU
        // warmup mid-pass AVs in ggml-sycl (locks can't prevent it — the SYCL
        // JIT/backend-init is an opaque call that ignores abort flags).  The UI
        // blocks the danger instead: can_reload() == false ⇒ the model/device
        // change is kept in Settings (already persisted above) but the reload is
        // refused until the in-flight load finishes; the user re-opens Settings
        // and hits OK again to apply it.  This keeps the scheduler simple — no
        // abort flag, no generation counter, no discard-and-restart command
        // buffer — because the load thread is NEVER interrupted.
        if (m_scheduler && (devChanged || modelChanged) && !m_modelPath.empty()) {
            if (!m_scheduler->can_reload()) {
                wxMessageBox(
                    "The model is still loading — please wait for it to finish "
                    "(the status bar shows Loading), then reopen Settings and "
                    "press OK to apply this change.\n\n"
                    "Switching models mid-load is blocked because interrupting "
                    "GPU warmup can crash the app.",
                    "Model Still Loading",
                    wxOK | wxICON_INFORMATION);
                return;  // keep the change in Settings; reload refused (safe)
            }
            m_scheduler->reload(m_modelPath, m_deviceIndex);
        } else if (m_modelPath.empty() && m_scheduler) {
            // Switched to no model — reload to an empty path drops the engine.
            // Still gate on can_reload() so we never interrupt an in-flight load.
            if (!m_scheduler->can_reload()) {
                wxMessageBox(
                    "The model is still loading — please wait for it to finish, "
                    "then reopen Settings to clear the model.",
                    "Model Still Loading",
                    wxOK | wxICON_INFORMATION);
                return;
            }
            m_scheduler->reload(m_modelPath, m_deviceIndex);
        }
    }
}

// ──────────────────────────────────────────
//  Event Handlers (stubs — no audio/engine)
// ──────────────────────────────────────────

void AppFrame::OnToggleRecord(wxCommandEvent& WXUNUSED(event)) {
    // CallAfter coalesces duplicate events into one.  Pure command layer: send
    // start/stop to the scheduler; RefreshUI (driven by the sync thread) owns
    // the button label/status text based on query_status().  No status code here.
    CallAfter([this]() {
        if (!m_recording) {
            // ── Start ──  If no model, ask for one; otherwise start() opens
            // PortAudio immediately (model-independent) and defers the pipeline
            // launch until warmup finishes — Record is instant even mid-load.
            if (!m_scheduler || m_modelPath.empty()) {
                wxMessageBox("Please load a model first (Settings).",
                             "No Model", wxOK | wxICON_INFORMATION);
                return;
            }
            // Block Record while the model is still loading — same gate as
            // Settings: the load thread isn't interruptible, and starting
            // capture mid-load (even though it buffers) is a confusing UX.
            // Wait for engine ready before allowing Record.
            if (!m_scheduler->can_reload()) {
                wxMessageBox(
                    wxString::FromUTF8("The model is still loading — please "
                        "wait for it to finish (status bar shows "
                        "\xe2\x80\x9cLoading model\xe2\x80\xa6\xe2\x80\x9d) "
                        "before starting recording."),
                    "Still Loading", wxOK | wxICON_INFORMATION);
                return;
            }
            if (!m_scheduler->start(m_micIndex)) {
                wxMessageBox(
                    "Could not open the microphone.\n\n"
                    "Common causes:\n"
                    "  - The device doesn't support 16 kHz capture (check Settings)\n"
                    "  - Another app holds the mic exclusively\n"
                    "  - No input device is connected",
                    "Microphone Error", wxOK | wxICON_WARNING);
                return;   // button stays "Record"; sync thread will reset state
            }
            m_recording.store(true);
            // The sync thread's next poll (≤100ms) flips the button to "Stop"
            // and the status to Recording/Loading-as-appropriate.
        } else {
            // ── Stop ──  The engine survives stop() for the next Record.
            if (m_scheduler) m_scheduler->stop();
            m_recording.store(false);
            // The sync thread flips the button back to "Record".
        }
    });
}

void AppFrame::OnClear(wxCommandEvent& WXUNUSED(event)) {
    m_transcriptText->Clear();
}

void AppFrame::OnCopy(wxCommandEvent& WXUNUSED(event)) {
    if (wxTheClipboard->Open()) {
        wxTheClipboard->SetData(new wxTextDataObject(m_transcriptText->GetValue()));
        wxTheClipboard->Close();
    }
}

void AppFrame::OnStatusBarClick(wxMouseEvent& ev) {
    wxStatusBar* sb = GetStatusBar();
    for (int i = 0; i < STATUS_FIELDS_COUNT; ++i) {
        wxRect r;
        sb->GetFieldRect(i, r);
        if (r.Contains(ev.GetPosition())) {
            if (i == STATUS_SETTINGS) {
                // Don't open Settings while a model is loading — switching
                // models mid-load can't be done safely (the load thread isn't
                // interruptible), and entering Settings then would only end in a
                // "still loading" refusal.  The status bar already shows
                // "Loading model…"; tell the user to wait instead of opening
                // the dialog.  (Record IS allowed during load — capture buffers
                // in the ring and the pipeline launches when warmup finishes.)
                if (m_scheduler && !m_scheduler->can_reload()) {
                    wxMessageBox(
                        wxString::FromUTF8("The model is still loading — please "
                            "wait for it to finish (status bar shows "
                            "\xe2\x80\x9cLoading model\xe2\x80\xa6\xe2\x80\x9d) "
                            "before opening Settings to switch the model or device."),
                        "Still Loading", wxOK | wxICON_INFORMATION);
                } else {
                    ShowSettingsDialog();
                }
            }
            break;
        }
    }
    ev.Skip();
}

void AppFrame::OnClose(wxCloseEvent& event) {
    // Stop the sync thread first so it stops polling before the scheduler is
    // torn down.  (~AppFrame also does this, but OnClose runs first.)
    m_stopSync.store(true);
    if (m_syncThread.joinable()) m_syncThread.join();
    // Stop any recording; then drop the scheduler (its dtor aborts + joins the
    // load thread and frees the engine).
    if (m_recording && m_scheduler)
        m_scheduler->stop();
    m_scheduler.reset();
    event.Skip();
}
