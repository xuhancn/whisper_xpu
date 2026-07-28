# whisper_xpu

Whisper speech recognition on Intel GPUs via SYCL (oneAPI), with oneDNN acceleration.

## Build from source

### Prerequisites

- **Windows 10/11** (Linux is cross-platform-targeted but not the verified path for this PR)
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) **2026.1+** — provides the SYCL compiler (`icx`/`icpx`). The SYCL core DLL delay-loads `sycl9.dll`; oneAPI 2025.x (which ships `sycl8.dll`) will not load.
- Visual Studio 2022 (Build Tools or IDE) — with the "Desktop development with C++" workload
- CMake **3.22+**
- Git (with submodules: whisper.cpp, wxWidgets, portaudio, oneDNN)

> The SYCL core compiles with the Intel oneAPI compiler; the GUI app and wxWidgets
> stay on MSVC. The Intel toolset name is supplied at configure time via the
> `DNNL_INTEL_TOOLSET` environment variable — no toolset version is hardcoded in
> CMake.

### Windows

1. Initialize the oneAPI + Intel toolset environment:

```bat
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2026"
```

2. Configure (Visual Studio 17 2022 generator, x64 — no global `-T`, so the app
   stays MSVC while SYCL targets pick up the Intel toolset per-target):

```bat
cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64 -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON
```

3. Build:

```bat
cmake --build build-gpu --config Release --parallel
```

To build only the headless pipeline test (faster — pulls in the SYCL core, whisper,
oneDNN, portaudio, but not wxWidgets):

```bat
cmake --build build-gpu --config Release --target test_pipeline --parallel
```

4. Output:

```
build-gpu\whisper-xpu-core\Release\whisper_xpu_sycl_core.dll  — SYCL core (engine, merge_segments, device_detect + ggml-sycl GPU ops + oneDNN, all in one DLL)
build-gpu\whisper-xpu-app\Release\whisper_xpu_app.exe         — wxWidgets GUI app
build-gpu\tests\streaming_pipeline\Release\test_pipeline.exe  — headless pipeline unit test
```

Runtime oneAPI DLLs (`sycl9.dll`, `ur_loader.dll`, …) are copied next to the
executables at build time ( guarded, so a missing optional DLL does not fail the build).

### Verify

CPU pipeline (regression-checked, 7/7 asserts pass):

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
build-gpu\tests\streaming_pipeline\Release\test_pipeline.exe --model models\ggml-tiny.bin --cpu
```

GPU (Intel Arc / Iris Xe, by SYCL device index):

```bat
build-gpu\tests\streaming_pipeline\Release\test_pipeline.exe --model models\ggml-tiny.bin --device 0
```

> **GPU path is currently non-functional at runtime — the build itself is green.**
> Under oneAPI 2026.1 the MSBuild linker (`lld-link`) ignores `-fsycl`, so the
> SPIR-V device image is not bundled into `whisper_xpu_sycl_core.dll` by the CMake
> build alone, and the Level Zero runtime then fails to resolve the first compute
> kernel: `No kernel named im2col_sycl<half> was found` (`Error OP IM2COL`,
> exit 1, 0/7 asserts run). This is a runtime/driver-level issue beyond the
> build layer — see PR #25. **CPU transcription is unaffected.**

### Linux (unverified in this PR)

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja
cmake --build build --config Release --parallel
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_SYCL` | `ON` | Enable the SYCL backend in whisper.cpp. Mandatory — configure fails if the oneAPI toolkit is not found. |
| `GGML_SYCL_DNN` | `ON` | Enable oneDNN in the SYCL backend. |
| `ONEDNN_STATIC` | `ON` | Build oneDNN from source as a static lib embedded into the SYCL core DLL. Set `OFF` to link the pre-built oneAPI oneDNN DLL. |

### Verify oneDNN linkage

```bat
dumpbin /dependents build-gpu\whisper-xpu-core\Release\whisper_xpu_sycl_core.dll | findstr dnnl
```

- **Empty output** — oneDNN is statically linked (`ONEDNN_STATIC=ON`, the default)
- **Shows `dnnl.dll`** — dynamically linked (`ONEDNN_STATIC=OFF`)

## Project structure

```
cmake/                      — FindSYCLToolkit.cmake, FindDNNL.cmake
whisper-xpu-core/           — SYCL core DLL (whisper_xpu_sycl_core.dll)
  sycl_src/                   — device detection + SYCL probe sources
whisper-xpu-app/            — wxWidgets GUI app + transcription scheduler
tests/streaming_pipeline/   — headless pipeline unit test (test_pipeline)
third_party/                — whisper.cpp, wxWidgets, portaudio, oneDNN
```
