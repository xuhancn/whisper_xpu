#include "app_frame.h"
#include "whisper_xpu_core.h"
#include "device_detect.h"
#include "audio_capture.h"
#include "src/audio_stream.h"

#include <wx/filedlg.h>
#include <wx/msgdlg.h>
#include <wx/statbox.h>
#include <wx/filename.h>

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

static whisper_xpu::Engine* safe_create_engine(const std::string& path, int device) {
    try {
        return new whisper_xpu::Engine(path, device);
    } catch (const std::exception&) {
        return nullptr;
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

    // Populate cached device/mic lists and update status bar.
    // SYCL device enumeration is deferred to OnIdleInit (after event
    // loop starts) via whisper_xpu_sycl_core.dll.  The DLL is
    // delay-loaded so the app starts regardless of sycl8.dll presence.
    m_micList = safe_enum_audio();
    UpdateStatusBar();

    // Load engine if a model was provided on the command line
    if (!m_modelPath.empty())
        LoadEngine(m_modelPath);

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

AppFrame::~AppFrame() {
    if (m_recording && m_audioStream)
        m_audioStream->stop();
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

    m_recordBtn = new wxToggleButton(panel, wxID_ANY, "Record");
    m_recordBtn->SetMinSize(wxSize(100, 32));
    m_recordBtn->SetToolTip("Start / stop recording (toggle)");
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
    m_recordBtn->Bind(wxEVT_TOGGLEBUTTON, &AppFrame::OnToggleRecord, this);
    m_copyBtn->Bind(wxEVT_BUTTON, &AppFrame::OnCopy, this);
    GetStatusBar()->Bind(wxEVT_LEFT_DOWN, &AppFrame::OnStatusBarClick, this);
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
    // Mic field — show actual device name, append (Default) if system default
    wxString micLabel = wxT("Mic: N/A");
    if (m_micIndex == kMicDefault) {
        for (const auto& m : m_micList) {
            if (m.is_default) {
                micLabel = wxT("Mic: ") + wxString(m.name) + wxT(" (Default)");
                break;
            }
        }
        if (micLabel == wxT("Mic: N/A") && !m_micList.empty()) {
            micLabel = wxT("Mic: ") + wxString(m_micList[0].name);
        }
    } else {
        for (const auto& m : m_micList) {
            if (m.index == m_micIndex) {
                micLabel = wxT("Mic: ") + wxString(m.name);
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
                devLabel = wxT("Device: ") + wxString(d.to_string());
                break;
            }
        }
    } else if (m_deviceIndex == kDeviceAuto) {
        // Auto-select: show first GPU if available
        for (const auto& d : m_deviceList) {
            if (d.index >= 0) {
                devLabel = wxT("Device: ") + wxString(d.to_string()) + wxT(" [Auto]");
                break;
            }
        }
    }
    SetStatusText(devLabel, STATUS_DEVICE);

    // Model field
    wxString modelLabel = "No model";
    if (!m_modelPath.empty()) {
        modelLabel = wxFileName(m_modelPath).GetFullName();
        if (m_engine) {
            modelLabel += m_engine->is_gpu_enabled()
                ? " [GPU]"
                : " [CPU]";
        }
    }
    SetStatusText(modelLabel, STATUS_MODEL);
}

bool AppFrame::LoadEngine(const std::string& path) {
    if (m_recording && m_audioStream) {
        m_audioStream->stop();
        m_recording = false;
        m_recordBtn->SetLabel("Record");
        m_recordBtn->SetValue(false);
    }
    m_engine.reset(safe_create_engine(path, m_deviceIndex));
    if (!m_engine) {
        wxMessageBox("Failed to initialize engine: SYCL runtime error.\n"
                     "Make sure the Intel GPU driver and oneAPI runtime are installed.",
                     "Error", wxOK | wxICON_ERROR);
        return false;
    }
    m_modelPath = path;
    UpdateStatusBar();
    return true;
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
    std::vector<wxString> devLabels;
    std::vector<int>       devIndices;
    devLabels.push_back("Auto");
    devIndices.push_back(kDeviceAuto);

    int devSel = 0;
    for (size_t i = 0; i < m_deviceList.size(); ++i) {
        wxString label(m_deviceList[i].to_string());
        if (label.Trim().IsEmpty()) continue;
        if (m_deviceList[i].index == m_deviceIndex) devSel = (int)devLabels.size();
        devLabels.push_back(label);
        devIndices.push_back((int)i);
    }

    auto* devBox = new wxStaticBoxSizer(wxHORIZONTAL, &dlg, "Device");
    auto* devChoice = new wxChoice(devBox->GetStaticBox(), wxID_ANY);
    for (size_t i = 0; i < devLabels.size(); ++i)
        devChoice->Append(devLabels[i]);
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

        bool devChanged   = (newDevIdx != m_deviceIndex);
        bool modelChanged = (newModelPath != m_modelPath);

        m_micIndex    = newMicIdx;
        m_deviceIndex = newDevIdx;
        m_modelPath   = newModelPath;
        m_hotkeyStr   = newHotkey;

        UpdateStatusBar();

        // Re-init engine if device or model changed
        if ((devChanged || modelChanged) && !m_modelPath.empty())
            LoadEngine(m_modelPath);
    }
}

// ──────────────────────────────────────────
//  Event Handlers (stubs — no audio/engine)
// ──────────────────────────────────────────

void AppFrame::OnToggleRecord(wxCommandEvent& WXUNUSED(event)) {
    if (m_recording) {
        // ── Stop recording ──
        if (m_audioStream)
            m_audioStream->stop();
        m_audioStream.reset();

        m_recordBtn->SetLabel("Record");
        m_recordBtn->SetValue(false);
        SetStatusText("Mic: Default", STATUS_MIC);
        m_recording = false;
    } else {
        // ── Start recording ──
        if (!m_engine) {
            wxMessageBox("Please load a model first (Settings).",
                         "No Model", wxOK | wxICON_INFORMATION);
            m_recordBtn->SetValue(false);
            return;
        }

        m_audioStream = std::make_unique<AudioStream>(m_engine.get(), m_transcriptText);
        m_audioStream->start(m_micIndex);

        m_recordBtn->SetLabel("Stop");
        m_recordBtn->SetValue(true);
        SetStatusText("Recording...", STATUS_MIC);
        m_recording = true;
    }
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
            if (i == STATUS_SETTINGS)
                ShowSettingsDialog();
            break;
        }
    }
    ev.Skip();
}

void AppFrame::OnClose(wxCloseEvent& event) {
    if (m_recording && m_audioStream)
        m_audioStream->stop();
    m_audioStream.reset();
    m_engine.reset();
    event.Skip();
}
