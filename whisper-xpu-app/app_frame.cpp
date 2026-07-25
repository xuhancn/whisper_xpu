#include "app_frame.h"
#include "whisper_xpu_core.h"
#include "device_detect.h"
#include "audio_capture.h"

#include <wx/filedlg.h>
#include <wx/dir.h>
#include <wx/msgdlg.h>
#include <wx/filename.h>
#include <mutex>
#include <sstream>
#include <vector>

wxBEGIN_EVENT_TABLE(AppFrame, wxFrame)
    EVT_CLOSE(AppFrame::OnClose)
wxEND_EVENT_TABLE()

AppFrame::AppFrame(const wxString& title, const wxPoint& pos, const wxSize& size,
                   const std::string& model_path, int device_index)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
    , m_modelPath(model_path)
    , m_deviceIndex(device_index)
    , m_micIndex(kMicDefault)
    , m_engine(nullptr)
    , m_audioCapture(nullptr)
{
    m_modelDir = wxGetCwd() + "/models";

    CreateControls();
    PopulateModelList();
    PopulateDeviceList();
    PopulateMicList();

    {
        auto devices = whisper_xpu::get_available_devices();
        std::ostringstream oss;
        oss << "=== Available Devices ===\n";
        for (const auto& d : devices) {
            oss << "  " << d.to_string() << "\n";
        }
        LogMessage(oss.str());
    }

    if (!m_modelPath.empty()) {
        LoadEngine(m_modelPath);
    }

    SetMinSize(wxSize(600, 400));
    SetSize(800, 600);
}

AppFrame::~AppFrame() {
    if (m_recording) {
        m_recording = false;
        if (m_audioThread.joinable()) {
            m_audioThread.join();
        }
    }
}

void AppFrame::CreateControls() {
    auto* mainPanel = new wxPanel(this, wxID_ANY);
    auto* topSizer = new wxBoxSizer(wxVERTICAL);

    // Model row
    auto* modelSizer = new wxBoxSizer(wxHORIZONTAL);
    modelSizer->Add(new wxStaticText(mainPanel, wxID_ANY, "Model:"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    m_modelChoice = new wxChoice(mainPanel, wxID_ANY);
    modelSizer->Add(m_modelChoice, 1, wxEXPAND | wxALL, 5);
    m_browseBtn = new wxButton(mainPanel, wxID_ANY, "Browse...");
    modelSizer->Add(m_browseBtn, 0, wxALL, 5);
    m_recordBtn = new wxToggleButton(mainPanel, wxID_ANY, "🎤  Start Recording");
    modelSizer->Add(m_recordBtn, 0, wxALL, 5);
    topSizer->Add(modelSizer, 0, wxEXPAND);

    // XPU Device row
    auto* deviceSizer = new wxBoxSizer(wxHORIZONTAL);
    deviceSizer->Add(new wxStaticText(mainPanel, wxID_ANY, "Device:"),
                     0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    m_deviceChoice = new wxChoice(mainPanel, ID_DEVICE_CHOICE);
    deviceSizer->Add(m_deviceChoice, 1, wxEXPAND | wxALL, 5);
    topSizer->Add(deviceSizer, 0, wxEXPAND);

    // Mic row
    auto* micSizer = new wxBoxSizer(wxHORIZONTAL);
    micSizer->Add(new wxStaticText(mainPanel, wxID_ANY, "Mic:"),
                   0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    m_micChoice = new wxChoice(mainPanel, ID_MIC_CHOICE);
    micSizer->Add(m_micChoice, 1, wxEXPAND | wxALL, 5);
    topSizer->Add(micSizer, 0, wxEXPAND);

    // Transcription output
    m_outputText = new wxTextCtrl(mainPanel, wxID_ANY, "",
                                  wxDefaultPosition, wxDefaultSize,
                                  wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH2);
    topSizer->Add(m_outputText, 3, wxEXPAND | wxALL, 5);

    // Log output
    auto* logLabel = new wxStaticText(mainPanel, wxID_ANY, "Log:");
    topSizer->Add(logLabel, 0, wxLEFT | wxRIGHT | wxTOP, 5);
    m_logText = new wxTextCtrl(mainPanel, wxID_ANY, "",
                               wxDefaultPosition, wxSize(-1, 120),
                               wxTE_MULTILINE | wxTE_READONLY);
    topSizer->Add(m_logText, 1, wxEXPAND | wxALL, 5);

    mainPanel->SetSizer(topSizer);
    m_statusBar = CreateStatusBar(2);

    m_browseBtn->Bind(wxEVT_BUTTON, &AppFrame::OnBrowseModel, this);
    m_recordBtn->Bind(wxEVT_TOGGLEBUTTON, &AppFrame::OnToggleRecord, this);
    m_deviceChoice->Bind(wxEVT_CHOICE, &AppFrame::OnSelectDevice, this);
    m_micChoice->Bind(wxEVT_CHOICE, &AppFrame::OnSelectMic, this);
    Bind(wxEVT_COMMAND_TEXT_UPDATED, [this](wxCommandEvent& ev) {
        if (ev.GetId() == ID_TRANSCRIBE_RESULT) {
            m_outputText->AppendText(ev.GetString());
        } else if (ev.GetId() == ID_TRANSCRIBE_ERROR) {
            LogMessage("Error: " + ev.GetString());
            SetRecording(false);
        }
    });
}

void AppFrame::PopulateModelList() {
    if (!wxDir::Exists(m_modelDir)) {
        LogMessage("Model directory not found: " + m_modelDir);
        LogMessage("Place model files (*.bin, *.ggml, *.gguf) in ./models/");
        return;
    }
    wxDir dir(m_modelDir);
    wxString filename;
    bool found = dir.GetFirst(&filename, "*.bin", wxDIR_FILES);
    if (!found) found = dir.GetFirst(&filename, "*.ggml", wxDIR_FILES);
    if (!found) found = dir.GetFirst(&filename, "*.gguf", wxDIR_FILES);
    while (found) {
        m_modelChoice->Append(filename);
        found = dir.GetNext(&filename);
    }
    if (m_modelChoice->GetCount() == 0) {
        LogMessage("No model files found in " + m_modelDir + ".");
    } else {
        m_modelChoice->SetSelection(0);
    }
}

void AppFrame::PopulateDeviceList() {
    auto devices = whisper_xpu::get_available_devices();
    m_deviceChoice->Clear();
    int sel = 0;
    for (size_t i = 0; i < devices.size(); ++i) {
        m_deviceChoice->Append(wxString(devices[i].to_string()));
        if (devices[i].index == m_deviceIndex)
            sel = static_cast<int>(i);
    }
    if (!devices.empty())
        m_deviceChoice->SetSelection(sel);
}

void AppFrame::PopulateMicList() {
    auto mics = AudioCapture::enumerate_devices();
    m_micChoice->Clear();
    m_micChoice->Append("System Default");
    int sel = 0;
    for (size_t i = 0; i < mics.size(); ++i) {
        m_micChoice->Append(wxString(mics[i].to_string()));
        if (mics[i].index == m_micIndex)
            sel = static_cast<int>(i + 1);
    }
    m_micChoice->SetSelection(sel);
}

void AppFrame::OnSelectModel(wxCommandEvent& WXUNUSED(event)) {
    int sel = m_modelChoice->GetSelection();
    if (sel == wxNOT_FOUND) return;
    wxString modelFile = m_modelChoice->GetString(sel);
    LoadEngine((wxFileName(m_modelDir, modelFile)).GetFullPath().ToStdString());
}

void AppFrame::OnBrowseModel(wxCommandEvent& WXUNUSED(event)) {
    wxFileDialog dialog(this, "Open Whisper Model File",
                        m_modelDir, "",
                        "Model files (*.bin;*.ggml;*.gguf)|*.bin;*.ggml;*.gguf|All files (*.*)|*.*",
                        wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    if (dialog.ShowModal() == wxID_CANCEL) return;
    LoadEngine(dialog.GetPath().ToStdString());
}

void AppFrame::OnSelectDevice(wxCommandEvent& WXUNUSED(event)) {
    auto devices = whisper_xpu::get_available_devices();
    int sel = m_deviceChoice->GetSelection();
    if (sel == wxNOT_FOUND || static_cast<size_t>(sel) >= devices.size()) return;
    int new_index = devices[sel].index;
    if (new_index != m_deviceIndex) {
        m_deviceIndex = new_index;
        LogMessage("Device changed to: " + wxString(devices[sel].to_string()));
        if (m_engine && !m_modelPath.empty()) {
            LoadEngine(m_modelPath);
        }
    }
}

void AppFrame::OnSelectMic(wxCommandEvent& WXUNUSED(event)) {
    auto mics = AudioCapture::enumerate_devices();
    int sel = m_micChoice->GetSelection();
    int new_mic = kMicDefault; // system default
    if (sel > 0 && static_cast<size_t>(sel - 1) < mics.size()) {
        new_mic = mics[sel - 1].index; // PortAudio device index
    }
    if (new_mic != m_micIndex) {
        m_micIndex = new_mic;
        if (new_mic >= 0) {
            LogMessage("Mic changed to: " + wxString(mics[sel - 1].to_string()));
        } else {
            LogMessage("Mic changed to: System Default");
        }
        if (m_recording) {
            SetRecording(false);
            SetRecording(true);
        }
    }
}

bool AppFrame::LoadEngine(const std::string& path) {
    if (m_recording) SetRecording(false);
    try {
        m_engine = std::make_unique<whisper_xpu::Engine>(path, m_deviceIndex);
        m_modelPath = path;
        std::ostringstream oss;
        oss << "Loaded: " << path;
        if (m_engine->is_gpu_enabled())
            oss << " [GPU: " << m_engine->device_description() << "]";
        else
            oss << " [CPU]";
        LogMessage(oss.str());
        m_statusBar->SetStatusText(m_engine->device_description());
        return true;
    } catch (const std::exception& e) {
        wxMessageBox("Failed to load model:\n" + wxString(e.what()),
                     "Error", wxOK | wxICON_ERROR);
        LogMessage("Error: " + wxString(e.what()));
        m_engine.reset();
        return false;
    }
}

void AppFrame::OnToggleRecord(wxCommandEvent& WXUNUSED(event)) {
    if (m_recording) {
        SetRecording(false);
    } else {
        if (!m_engine) {
            wxMessageBox("No model loaded. Select a model first.",
                         "Error", wxOK | wxICON_WARNING);
            m_recordBtn->SetValue(false);
            return;
        }
        SetRecording(true);
    }
}

void AppFrame::SetRecording(bool active) {
    if (active == m_recording) return;

    if (active) {
        m_recording = true;
        m_recordBtn->SetLabel("⏹  Stop Recording");
        m_recordBtn->SetValue(true);
        m_statusBar->SetStatusText("🔴 Recording...");
        m_outputText->Clear();
        LogMessage("Recording started...");

        auto sampleBuf = std::make_shared<std::vector<float>>();
        auto bufMutex  = std::make_shared<std::mutex>();

        m_audioCapture = std::make_unique<AudioCapture>();
        m_audioCapture->set_callback(
            [sampleBuf, bufMutex](const float* samples, size_t count) -> size_t {
                std::lock_guard<std::mutex> lock(*bufMutex);
                sampleBuf->insert(sampleBuf->end(), samples, samples + count);
                return count;
            }
        );

        if (!m_audioCapture->start(m_micIndex)) {
            LogMessage("Error: Failed to start microphone capture.");
            m_recording = false;
            m_recordBtn->SetLabel("🎤  Start Recording");
            m_recordBtn->SetValue(false);
            m_statusBar->SetStatusText("Microphone error");
            return;
        }

        m_audioThread = std::thread([this, sampleBuf, bufMutex]() {
            while (m_recording) {
                std::vector<float> chunk;
                {
                    std::lock_guard<std::mutex> lock(*bufMutex);
                    if (sampleBuf->size() >= 16000) {
                        chunk.assign(sampleBuf->begin(), sampleBuf->end());
                        sampleBuf->clear();
                    }
                }
                if (!chunk.empty() && m_engine) {
                    auto result = m_engine->transcribe_stream(
                        [&chunk](float* buf, size_t max) -> size_t {
                            size_t n = std::min(chunk.size(), max);
                            std::copy(chunk.begin(), chunk.begin() + n, buf);
                            chunk.erase(chunk.begin(), chunk.begin() + n);
                            return n;
                        }
                    );
                    if (!result.text.empty()) {
                        wxQueueEvent(this, new wxThreadEvent(
                            wxEVT_COMMAND_TEXT_UPDATED, ID_TRANSCRIBE_RESULT));
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    } else {
        m_recording = false;
        if (m_audioThread.joinable()) m_audioThread.join();
        if (m_audioCapture) {
            m_audioCapture->stop();
            m_audioCapture.reset();
        }
        m_recordBtn->SetLabel("🎤  Start Recording");
        m_recordBtn->SetValue(false);
        m_statusBar->SetStatusText("Idle");
        LogMessage("Recording stopped.");
    }
}

void AppFrame::OnClose(wxCloseEvent& event) {
    if (m_recording) {
        m_recording = false;
        if (m_audioThread.joinable()) m_audioThread.join();
    }
    m_engine.reset();
    m_audioCapture.reset();
    event.Skip();
}

void AppFrame::LogMessage(const wxString& text) {
    m_logText->AppendText(text + "\n");
    int lastNl = text.Find('\n', true);
    if (lastNl != wxNOT_FOUND)
        m_statusBar->SetStatusText(text.Mid(lastNl + 1));
    else if (!text.empty())
        m_statusBar->SetStatusText(text);
}
