# whisper_xpu

Whisper speech recognition on Intel GPUs via SYCL (oneAPI), with oneDNN acceleration.

## Build from source

### Prerequisites

- Windows 10/11 or Linux
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (2025.x+) — provides the SYCL compiler
- Visual Studio 2022 (Windows) — with "Desktop development with C++" workload
- CMake 3.20+
- Git

### Windows

1. Open **x64 Native Tools Command Prompt for VS 2022** (or run `vcvars64.bat`)

2. Activate oneAPI environment and configure:

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cmake -S . -B build-sycl
```

3. Build:

```bat
cmake --build build-sycl --config Release --parallel
```

4. Output:

```
build-sycl\bin\Release\ggml-sycl.dll    — SYCL backend with oneDNN
build-sycl\whisper-xpu-app\Release\whisper_xpu_app.exe  — GUI app
```

### Linux

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja
cmake --build build --config Release --parallel
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `ONEDNN_STATIC` | `ON` | Build oneDNN from source as static lib (no `dnnl.dll` needed). Set `OFF` to use pre-built oneAPI oneDNN DLL. |
| `WHISPER_XPU_USE_GPU` | auto | Enable GPU acceleration via SYCL. Defaults to ON when oneAPI toolkit is detected. |

### Verify oneDNN linkage

```bat
dumpbin /dependents build-sycl\bin\Release\ggml-sycl.dll | findstr dnnl
```

- **Empty output** — oneDNN is statically linked (ONEDNN_STATIC=ON, the default)
- **Shows `dnnl.dll`** — dynamically linked (ONEDNN_STATIC=OFF)

## Project structure

```
cmake/                  — FindSYCLToolkit.cmake, FindDNNL.cmake
whisper-xpu-core/       — Core engine library (static)
whisper-xpu-app/        — wxWidgets GUI application
third_party/            — whisper.cpp, wxWidgets, portaudio, oneDNN
```
