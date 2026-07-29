# whisper_xpu

Whisper speech recognition on Intel GPUs via SYCL (oneAPI), with oneDNN acceleration.

## Build from source

### Prerequisites

- **Windows 10/11** (Linux is cross-platform-targeted but not the verified path for this PR)
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) **2025.3** (tested with 2025.3.x) — provides the SYCL compiler (`icx`/`icpx`) and the `sycl8.dll` runtime. **Do not use 2026.1**: under 2026.1 the Level Zero runtime fails to resolve the first compute kernel (`No kernel named im2col_sycl<half>`); pin 2025.3 for a working GPU path.
- [Ninja](https://ninja-build.org/) 1.11+ — `winget install Ninja-build.Ninja` (or `choco install ninja`). The SYCL sub-build is driven by the Ninja `icx` link (see the Windows section below).
- Visual Studio 2022 (Build Tools or IDE) — with the "Desktop development with C++" workload
- CMake **3.22+**
- Git (with submodules: whisper.cpp, wxWidgets, portaudio, oneDNN)

> The SYCL core's GPU kernels are compiled by an `icx` + Ninja *sub-build* of
> whisper.cpp (driven by `ExternalProject`), which bundles the SPIR-V device
> image into `ggml-sycl.dll`. The GUI app, wxWidgets, portaudio, and the thin
> SYCL-core wrapper stay on MSVC. Under the Ninja generator oneDNN is built with
> `icx` directly (no VS toolset); the Visual-Studio-generator alternative uses
> `DNNL_INTEL_TOOLSET`.

### Windows (PowerShell + Ninja — recommended)

1. Initialize the **2025.3** oneAPI environment. The per-version
   `oneapi-vars.bat` pins 2025.3 — the root `setvars.bat` would load 2026.1,
   which is broken for the GPU path:

```powershell
$env:VS2022INSTALLDIR = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
& "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat"
```

2. Configure + build. Force `cl` as the **top-level** compiler so wxWidgets
   (which rejects `icx` as a WIN32 compiler with `Unknown WIN32 compiler type`)
   builds on MSVC; the whisper.cpp and oneDNN sub-builds still use `icx` — they
   set `-DCMAKE_C_COMPILER=icx` themselves. Ninja is single-config, so `Release`
   is fixed at configure time (`--config` is accepted but ignored). SYCL + oneDNN
   are enabled by the sub-build by default; `GGML_SYCL`/`GGML_SYCL_DNN` are not
   top-level options (passing them is a harmless no-op — CMake reports them
   unused):

```powershell
cmake -S . -B build-gpu -G Ninja `
      -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=cl `
      -DCMAKE_CXX_COMPILER=cl
cmake --build build-gpu --parallel
```

   Headless-only (faster — SYCL core + whisper + oneDNN + portaudio, no wx):

```powershell
cmake --build build-gpu --target test_pipeline --parallel
```

3. Run the GPU pipeline. Single-config Ninja puts exes directly under each
   target dir (no `Release\` subfolder):

```powershell
.\build-gpu\tests\streaming_pipeline\test_pipeline.exe `
    --device 0 --model models\ggml-tiny.bin
```

   `--device 0` selects the first SYCL device (Intel Arc / Iris Xe); use `--cpu`
   for the CPU regression path. Re-run the `oneapi-vars.bat` line first if any
   runtime DLL was not copied beside the exe.

4. Output + runtime DLLs. The build co-locates beside each executable:

- `whisper_xpu_sycl_core.dll` — SYCL core (engine, merge_segments, device_detect)
- `whisper.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll` — CPU/backend chain
- `ggml-sycl.dll` (~120 MB, SPIR-V device image bundled by the `icx`/Ninja link)
- oneAPI runtime (guarded copy of whatever exists under 2025.3): `sycl8.dll`,
  `ur_loader.dll`, `ur_adapter_level_zero.dll`, `libmmd.dll`, `libiomp5md.dll`,
  `tcm.dll`, …

   Any oneAPI DLL not copied is resolved via the `oneapi-vars.bat` `PATH`. For a
   clean run without oneAPI sourced, copy the missing DLL from
   `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\` next to the exe.

### Windows — alternative (Visual Studio generator)

A Visual-Studio-generator top-level build also works; the untracked
`build_gpu.bat` is a ready-made repro script. With the VS generator, pass the
Intel toolset via `DNNL_INTEL_TOOLSET` and pin 2025.3 the same way:

```bat
set "VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat"
set "DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025"
cmake -S . -B build-gpu -G "Visual Studio 17 2022" -A x64
cmake --build build-gpu --config Release --parallel
```

Multi-config VS puts exes under a `Release\` subfolder, e.g.
`build-gpu\tests\streaming_pipeline\Release\test_pipeline.exe`. Output also
includes `build-gpu\whisper-xpu-app\Release\whisper_xpu_app.exe` (wxWidgets GUI).

### Verify

CPU pipeline (regression-checked, 7/7 asserts pass):

```powershell
& "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat"
.\build-gpu\tests\streaming_pipeline\test_pipeline.exe --model models\ggml-tiny.bin --cpu
```

GPU (Intel Arc / Iris Xe, by SYCL device index):

```powershell
.\build-gpu\tests\streaming_pipeline\test_pipeline.exe --model models\ggml-tiny.bin --device 0
```

> **GPU path is functional** — 7/7 pipeline asserts pass on Arc A770M with
> oneAPI 2025.3 and the `icx`/Ninja sub-build, which bundles the SPIR-V device
> image into `ggml-sycl.dll`. The earlier 2026.1 failure (`No kernel named
> im2col_sycl<half>`, caused by `lld-link` dropping `-fsycl`) is avoided by
> pinning 2025.3. CPU transcription (`--cpu`) is unaffected either way.

### Linux (unverified in this PR)

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja
cmake --build build --config Release --parallel
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `GGML_SYCL` | `ON` (sub-build) | SYCL backend in the whisper.cpp sub-build. Set internally; not a top-level option (FindSYCLToolkit makes oneAPI mandatory). |
| `GGML_SYCL_DNN` | `ON` (sub-build) | oneDNN in the SYCL backend. Set internally by the sub-build. |
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
