#include "app_frame.h"
#include "engine.h"
#include "device_detect.h"
#include "audio_capture.h"

#include <wx/filedlg.h>
#include <wx/dir.h>
#include <wx/msgdlg.h>
#include <wx/filename.h>
#include <mutex>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------
// Event table
// ---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(AppFrame, wxFrame)
    EVT_CLOSE(AppFrame::OnClose)
wxEND_EVENT_TABLE()

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

AppFrame::AppFrame(const wxString& title, const wxPoint& pos, const wxSize& size,
                   const std::string& model_path, bool use_gpu)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
    , m_modelPath(model_path)
    , m_useGpu(use_gpu)
    , m_engine(nullptr)
    , m_audioCapture(nullptr)
{
    m_modelDir = wxGetCwd() + "/models";

    CreateControls();
    PopulateModelList();

    // Detect and log available devices
    {
        auto devices = detect_devices();
        std::ostringstream oss;
        oss << "=== Available Devices ===\n";
        for (const auto& d : devices) {
            oss << "  " << d << "\n";
        }
        LogMessage(oss.str());
    }

    LogMessage("Best device: " + wxString(get_best_device_name()));

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

// ---------------------------------------------------------------------------
// UI Creation
// ---------------------------------------------------------------------------

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

    // Status bar
    m_statusBar = CreateStatusBar(2);

    // Bind events
    m_browseBtn->Bind(wxEVT_BUTTON, &AppFrame::OnBrowseModel, this);
    m_recordBtn->Bind(wxEVT_TOGGLEBUTTON, &AppFrame::OnToggleRecord, this);
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

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

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

bool AppFrame::LoadEngine(const std::string& path) {
    if (m_recording) SetRecording(false);
    try {
        m_engine = std::make_unique<whisper_xpu::Engine>(path, m_useGpu);
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

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

void AppFrame::SetRecording(bool active) {
    if (active == m_recording) return;

    if (active) {
        // Start
        m_recording = true;
        m_recordBtn->SetLabel("⏹  Stop Recording");
        m_recordBtn->SetValue(true);
        m_statusBar->SetStatusText("🔴 Recording...");
        m_outputText->Clear();
        LogMessage("Recording started...");

        // Shared sample buffer (thread-safe via mutex)
        auto sampleBuf = std::make_shared<std::vector<float>>();
        auto bufMutex  = std::make_shared<std::mutex>();

        // Set up capture callback: PortAudio thread pushes samples
        m_audioCapture = std::make_unique<AudioCapture>();
        m_audioCapture->set_callback(
            [sampleBuf, bufMutex](const float* samples, size_t count) -> size_t {
                std::lock_guard<std::mutex> lock(*bufMutex);
                sampleBuf->insert(sampleBuf->end(), samples, samples + count);
                return count;
            }
        );

        if (!m_audioCapture->start()) {
            LogMessage("Error: Failed to start microphone capture.");
            m_recording = false;
            m_recordBtn->SetLabel("🎤  Start Recording");
            m_recordBtn->SetValue(false);
            m_statusBar->SetStatusText("Microphone error");
            return;
        }

        // Processing thread: periodically drain the buffer and run inference
        m_audioThread = std::thread([this, sampleBuf, bufMutex]() {
            while (m_recording) {
                std::vector<float> chunk;
                {
                    std::lock_guard<std::mutex> lock(*bufMutex);
                    if (sampleBuf->size() >= 16000) { // at least 1 second
                        chunk.assign(sampleBuf->begin(), sampleBuf->end());
                        sampleBuf->clear();
                    }
                }

                if (!chunk.empty() && m_engine) {
                    // Run inference
                    auto result = m_engine->transcribe_stream(
                        [&chunk](float* buf, size_t max) -> size_t {
                            size_t n = std::min(chunk.size(), max);
                            std::copy(chunk.begin(), chunk.begin() + n, buf);
                            chunk.erase(chunk.begin(), chunk.begin() + n);
                            return n;
                        }
                    );

                    if (!result.text.empty()) {
                        // Post result text to UI thread
                        wxQueueEvent(this, new wxThreadEvent(
                            wxEVT_COMMAND_TEXT_UPDATED, ID_TRANSCRIBE_RESULT));
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    } else {
        // Stop
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
