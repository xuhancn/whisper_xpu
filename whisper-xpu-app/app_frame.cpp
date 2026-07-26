#include "app_frame.h"
#include "whisper_xpu_core.h"
#include "device_detect.h"
#include "audio_capture.h"

#include <wx/filedlg.h>
#include <wx/msgdlg.h>
#include <wx/statbox.h>

wxBEGIN_EVENT_TABLE(AppFrame, wxFrame)
    EVT_CLOSE(AppFrame::OnClose)
wxEND_EVENT_TABLE()

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

    wxFont emojiFont(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL,
                     wxFONTWEIGHT_NORMAL, false, "Segoe UI Emoji");

    m_clearBtn = new wxButton(panel, wxID_ANY, "🗑  Clear");
    m_clearBtn->SetFont(emojiFont);
    m_clearBtn->SetToolTip("Clear all transcription text");
    btnSizer->Add(m_clearBtn, 0, wxRIGHT, 8);

    m_recordBtn = new wxToggleButton(panel, wxID_ANY, "⏺  Record");
    m_recordBtn->SetFont(emojiFont);
    m_recordBtn->SetToolTip("Start / stop recording (toggle)");
    btnSizer->Add(m_recordBtn, 0, wxRIGHT, 8);

    m_copyBtn = new wxButton(panel, wxID_ANY, "📋  Copy");
    m_copyBtn->SetFont(emojiFont);
    m_copyBtn->SetToolTip("Copy transcription to clipboard");
    btnSizer->Add(m_copyBtn, 0);

    root->Add(btnSizer, 0, wxALIGN_CENTER | wxBOTTOM, 10);

    panel->SetSizer(root);

    // ── 4-segment status bar ──
    CreateStatusBarFields();

    // ── Events ──
    m_clearBtn->Bind(wxEVT_BUTTON, &AppFrame::OnClear, this);
    m_recordBtn->Bind(wxEVT_TOGGLEBUTTON, &AppFrame::OnToggleRecord, this);
    m_copyBtn->Bind(wxEVT_BUTTON, &AppFrame::OnCopy, this);
    GetStatusBar()->Bind(wxEVT_LEFT_DOWN, &AppFrame::OnStatusBarClick, this);
}

void AppFrame::CreateStatusBarFields() {
    CreateStatusBar(STATUS_FIELDS_COUNT);

    // Widths: negative = stretch proportional, positive = fixed px
    int widths[STATUS_FIELDS_COUNT] = { -1, 130, -1, 50 };
    SetStatusWidths(STATUS_FIELDS_COUNT, widths);

    SetStatusText("Mic: Default",  STATUS_MIC);
    SetStatusText("Device: Auto",  STATUS_DEVICE);
    SetStatusText("No model",      STATUS_MODEL);
    SetStatusText("...",           STATUS_SETTINGS);
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
    // TODO: populate from AudioCapture::enumerate_devices()
    micChoice->SetSelection(0);
    micBox->Add(micChoice, 1, wxEXPAND | wxALL, 5);
    root->Add(micBox, 0, wxEXPAND | wxALL, 8);

    // ── XPU Device ──
    auto* devBox = new wxStaticBoxSizer(wxHORIZONTAL, panel, "Device");
    auto* devChoice = new wxChoice(panel, wxID_ANY);
    devChoice->Append("Auto");
    // TODO: populate from whisper_xpu::get_available_devices()
    devChoice->SetSelection(0);
    devBox->Add(devChoice, 1, wxEXPAND | wxALL, 5);
    root->Add(devBox, 0, wxEXPAND | wxALL, 8);

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
    root->Add(modelBox, 0, wxEXPAND | wxALL, 8);

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
    root->Add(hotkeyBox, 0, wxEXPAND | wxALL, 8);

    root->AddStretchSpacer();

    // ── OK / Cancel ──
    auto* dlgBtns = dlg.CreateButtonSizer(wxOK | wxCANCEL);
    root->Add(dlgBtns, 0, wxALIGN_RIGHT | wxALL, 10);

    panel->SetSizer(root);

    dlg.SetMinSize(wxSize(440, 400));
    root->SetSizeHints(&dlg);

    if (dlg.ShowModal() == wxID_OK) {
        // Stub — save settings for future wiring
        // m_micIndex    = ...;
        // m_deviceIndex = ...;
        // m_modelPath   = modelText->GetValue().ToStdString();
        // m_hotkeyStr   = hotkeyText->GetValue();
    }
}

// ──────────────────────────────────────────
//  Event Handlers (stubs — no audio/engine)
// ──────────────────────────────────────────

void AppFrame::OnToggleRecord(wxCommandEvent& WXUNUSED(event)) {
    if (m_recording) {
        // ── Stop recording ──
        m_recordBtn->SetLabel("⏺  Record");
        m_recordBtn->SetValue(false);
        SetStatusText("Mic: Default", STATUS_MIC);

        m_recording = false;
        if (m_audioThread.joinable())
            m_audioThread.join();
    } else {
        // ── Start recording ──
        // TODO: check engine loaded, start AudioCapture, spawn thread
        m_recordBtn->SetLabel("⏹  Stop");
        m_recordBtn->SetValue(true);
        SetStatusText("🔴 Recording...", STATUS_MIC);

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
