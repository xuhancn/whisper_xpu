#include "app_frame.h"
#include "whisper_xpu_core.h"
#include "device_detect.h"
#include "audio_capture.h"

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

// ── Thin wrappers with error handling ──
// SYCL device enumeration (get_available_devices) is intentionally
// NOT called here — the Intel GPU driver / Level Zero version on
// this system can cause an AV inside sycl8.dll that can't be caught
// from MSVC code (C2712 prevents __try/__except with vector).  The
// Engine constructor handles device auto-selection internally.

static std::vector<whisper_xpu::DeviceInfo> safe_get_devices() {
    // sycl::platform::get_platforms() inside sycl8.dll can AV with
    // a null function pointer on certain Intel GPU driver / Level Zero
    // version combinations.  We skip enumeration entirely and let the
    // Engine auto-select a device when a model is loaded.
    //
    // The settings dialog shows only "Auto" when the list is empty.
    return {};
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
    // SYCL init is deferred to OnIdleInit (after event loop starts) to
    // avoid a null-function-pointer crash inside sycl8.dll's platform
    // enumeration when the Intel GPU driver version doesn't match the
    // oneAPI runtime.  Audio enumeration is safe — no SYCL dependency.
    m_micList = safe_enum_audio();
    UpdateStatusBar();

    // Load engine if a model was provided on the command line
    if (!m_modelPath.empty())
        LoadEngine(m_modelPath);

    // Delay SYCL device detection until the event loop is running
    Bind(wxEVT_IDLE, &AppFrame::OnIdleInit, this);
}

void AppFrame::OnIdleInit(wxIdleEvent& event) {
    // Run once: unbind immediately so we only fire once.
    Unbind(wxEVT_IDLE, &AppFrame::OnIdleInit, this);

    // SYCL platform enumeration is skipped (safe_get_devices returns
    // empty).  The Engine auto-selects the device when a model loads.
    // If the device list is ever non-empty here, update the status bar.
    UpdateStatusBar();

    event.Skip();
}

AppFrame::~AppFrame() {
    if (m_recording) {
        m_recording = false;
        if (m_audioThread.joinable())
            m_audioThread.join();
    }
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
    // Mic field
    wxString micLabel = "Mic: Default";
    if (m_micIndex == kMicDefault) {
        micLabel = "Mic: Default";
    } else {
        for (const auto& m : m_micList) {
            if (m.index == m_micIndex) {
                micLabel = "Mic: " + wxString(m.name);
                break;
            }
        }
    }
    SetStatusText(micLabel, STATUS_MIC);

    // Device field
    wxString devLabel = "Device: Auto";
    if (m_deviceIndex == kDeviceCPU) {
        devLabel = "Device: CPU";
    } else if (m_deviceIndex >= 0) {
        for (const auto& d : m_deviceList) {
            if (d.index == m_deviceIndex) {
                devLabel = "Device: " + wxString(d.to_string());
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
    if (m_recording) {
        m_recording = false;
        if (m_audioThread.joinable())
            m_audioThread.join();
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
    wxDialog dlg(this, wxID_ANY, "Settings", wxDefaultPosition, wxSize(440, 400));
    auto* panel = new wxPanel(&dlg);
    auto* root  = new wxBoxSizer(wxVERTICAL);

    // ── Microphone ──
    auto* micBox = new wxStaticBoxSizer(wxHORIZONTAL, panel, "Microphone");
    auto* micChoice = new wxChoice(panel, wxID_ANY);
    micChoice->Append("System Default");
    {
        int sel = 0;
        for (size_t i = 0; i < m_micList.size(); ++i) {
            micChoice->Append(wxString(m_micList[i].name));
            if (m_micList[i].index == m_micIndex)
                sel = static_cast<int>(i + 1);
        }
        micChoice->SetSelection(sel);
    }
    micBox->Add(micChoice, 1, wxEXPAND | wxALL, 5);
    root->Add(micBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    // ── XPU Device ──
    auto* devBox = new wxStaticBoxSizer(wxHORIZONTAL, panel, "Device");
    auto* devChoice = new wxChoice(panel, wxID_ANY);
    devChoice->Append("Auto");
    {
        int sel = 0;
        for (size_t i = 0; i < m_deviceList.size(); ++i) {
            devChoice->Append(wxString(m_deviceList[i].to_string()));
            if (m_deviceList[i].index == m_deviceIndex)
                sel = static_cast<int>(i + 1);
        }
        devChoice->SetSelection(sel);
    }
    devBox->Add(devChoice, 1, wxEXPAND | wxALL, 5);
    root->Add(devBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    // ── Model path + browse ──
    auto* modelBox = new wxStaticBoxSizer(wxHORIZONTAL, panel, "Model");
    auto* modelText = new wxTextCtrl(panel, wxID_ANY, wxString(m_modelPath),
                                     wxDefaultPosition, wxDefaultSize,
                                     wxTE_PROCESS_ENTER);
    modelText->SetToolTip("Path to a Whisper model file (.bin / .ggml / .gguf)");
    modelBox->Add(modelText, 1, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    auto* browseBtn = new wxButton(panel, wxID_ANY, "Browse...");
    browseBtn->Bind(wxEVT_BUTTON, [&dlg, modelText](wxCommandEvent&) {
        wxFileDialog fd(&dlg, "Open Whisper Model File", "", "",
                        "Model files (*.bin;*.ggml;*.gguf)|*.bin;*.ggml;*.gguf"
                        "|All files (*.*)|*.*",
                        wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (fd.ShowModal() == wxID_OK)
            modelText->SetValue(fd.GetPath());
    });
    modelBox->Add(browseBtn, 0, wxALL, 5);
    root->Add(modelBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    // ── Record Hotkey ──
    auto* hotkeyBox = new wxStaticBoxSizer(wxHORIZONTAL, panel, "Record Hotkey");
    auto* hotkeyText = new wxTextCtrl(panel, wxID_ANY, m_hotkeyStr,
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
            return; // ignore modifier-only or unsupported keys
        }
        hotkeyText->SetValue(combo);
    });
    hotkeyBox->Add(hotkeyText, 0, wxALL, 5);
    hotkeyBox->Add(new wxStaticText(panel, wxID_ANY, "(Click field, press key)"),
                   0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    root->Add(hotkeyBox, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    root->AddStretchSpacer();

    // ── OK / Cancel ──
    auto* dlgBtns = dlg.CreateButtonSizer(wxOK | wxCANCEL);
    root->Add(dlgBtns, 0, wxALIGN_RIGHT | wxBOTTOM | wxRIGHT, 10);

    // Panel fills the dialog; sizer lays out the controls inside the panel.
    panel->SetSizer(root);

    auto* dlgSizer = new wxBoxSizer(wxVERTICAL);
    dlgSizer->Add(panel, 1, wxEXPAND);
    dlg.SetSizer(dlgSizer);

    dlg.SetMinSize(wxSize(440, 400));
    dlgSizer->SetSizeHints(&dlg);

    if (dlg.ShowModal() == wxID_OK) {
        // ── Read mic selection ──
        int newMicIdx = kMicDefault;
        {
            int sel = micChoice->GetSelection();
            if (sel > 0 && static_cast<size_t>(sel - 1) < m_micList.size())
                newMicIdx = m_micList[sel - 1].index;
        }

        // ── Read device selection ──
        int newDevIdx = kDeviceAuto;
        {
            int sel = devChoice->GetSelection();
            if (sel > 0 && static_cast<size_t>(sel - 1) < m_deviceList.size())
                newDevIdx = m_deviceList[sel - 1].index;
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
        m_recordBtn->SetLabel("Record");
        m_recordBtn->SetValue(false);
        SetStatusText("Mic: Default", STATUS_MIC);

        m_recording = false;
        if (m_audioThread.joinable())
            m_audioThread.join();
    } else {
        // ── Start recording ──
        // TODO: check engine loaded, start AudioCapture, spawn thread
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
    if (m_recording) {
        m_recording = false;
        if (m_audioThread.joinable())
            m_audioThread.join();
    }
    m_engine.reset();
    m_audioCapture.reset();
    event.Skip();
}
