# whisper_xpu

[English](README.md) | **中文**

> ### **在 Intel GPU 上做实时语音转文字，基于 SYCL（oneAPI）+ oneDNN。**

`whisper_xpu` 在 Intel Arc / Iris Xe GPU 上运行 OpenAI 的 Whisper 模型，借助
Intel 的 SYCL 运行时，配一个实时流式转写的 wxWidgets 桌面应用 ——
这是迈向 **语音驱动的编码工作流** 的基础。

> **关于老款 Intel 显卡：** Arc（独显）和 Iris Xe（核显）可用。更老的 Intel
> UHD / HD Graphics **不支持** —— 缺少 ggml-sycl 内核所需的 SYCL / Level Zero
> 特性。CPU 路径在任何机器上都能跑，与 GPU 无关。

![whisper_xpu](Images/app.png)

---

## 关于本项目

**现在** —— 一个 Windows 桌面应用，持续采集麦克风音频，切成 5 秒一个窗口，
每个窗口到达即在 GPU 上转写 —— 文字逐字实时出现，带重叠去重，边界处不丢字、
不重复。计算由 whisper.cpp 的 SYCL backend 完成；oneDNN 加速 GEMM；调度器跑一个
小型 worker 池，让采集和转写解耦（模型还在加载时就能开始录音）。

**方向** —— 一个语音驱动的编码 agent / 工作流：自然说话，让转写结果驱动编码动作，
而不是只落进一个文本框。这里的流式转写引擎是基底；更长远的规划是把这条流接进
编码循环（起草 → 编辑 → 运行 → 迭代），全程语音。当前发布是这条路上的
"说话 → 稳定的实时文字" 这一步。

中文输出可以即时渲染成简体或繁体字形（内置 OpenCC 词典），无论 Whisper 原本
输出的是哪种字形。

---

## 快速开始

### 方式 A — 下载发布包（最简单）

从 [latest release](https://github.com/xuhancn/whisper_xpu/releases/latest) 下载
**`whisper_xpu_v1.0.0.zip`**。解压 → 双击 `whisper_xpu_app.exe` → 即可使用
（无需 oneAPI 环境、无需编译）。包内自带 `ggml-tiny.bin`，默认用 **CPU**，
所以在任何 Windows 11 机器上都能跑；打开 **Settings**（点击状态栏）可选更大的
模型或 Intel Arc GPU 以获得快得多的转写。

目标机器要求：Windows 10/11 x64、一个麦克风，以及 —— 仅在要用 GPU 时 ——
足够新、能提供 `ze_loader.dll` 的 Intel Arc / Iris Xe 驱动（即 Intel GPU 驱动，
位于 System32）。CPU 模式无需额外依赖。

### 方式 B — 从源码构建

下面只记录 **已验证** 的构建路径（Ninja + oneAPI 2025.3，Windows）。其它
生成器/平台试过但不可靠，这里不保留。

---

## 模型

发布 zip **只** 自带 `ggml-tiny.bin`（约 77 MB）—— 足够开箱即用，且 tiny 在 CPU 上
很快。真正使用时你会想要更大、更准的模型；自行下载到 `models/`，然后在 Settings
（或 `whisper_xpu.ini`）里指向它。

`q5_0`（5-bit）量化模型是已验证、推荐的集合 —— 在 CPU 和 Intel GPU 上都能用。
从官方 [whisper.cpp HuggingFace 仓库](https://huggingface.co/ggerganov/whisper.cpp)
（`ggerganov/whisper.cpp`）下载：

| 模型 | 文件 | 大小 | 已验证 |
|---|---|---|---|
| tiny | `ggml-tiny.bin` | ~77 MB | ✓（发布包自带）|
| medium | `ggml-medium-q5_0.bin` | ~540 MB | ✓（CPU + GPU）|
| large-v3 | `ggml-large-v3-q5_0.bin` | ~1.0 GB | ✓（CPU + GPU）|
| large-v3 turbo | `ggml-large-v3-turbo-q5_0.bin` | ~570 MB | ✓（CPU + GPU）|

下载（任选其一）：

```powershell
# 例如 turbo —— 更快且够准：
curl -L -o models/ggml-large-v3-turbo-q5_0.bin `
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
# 或完整的 large-v3（最准，最慢）：
curl -L -o models/ggml-large-v3-q5_0.bin `
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin
```

然后设为默认 —— 要么打开 Settings 选它，要么编辑 `whisper_xpu.ini`：

```ini
model=models/ggml-large-v3-turbo-q5_0.bin
```

> **在 GPU 上优先用 `q5_0` 而非 `q8_0`。** `q8_0` 模型在 Intel Arc GPU 上有问题
> （问题在上游 ggml-sycl 的算子）；`q5_0` 在 CPU 和 GPU 上都正常。

> **GPU 在 worker 触碰前要先预热。** SYCL 的首次内核 JIT + 每 state buffer 分配
> 在 **load 线程**（`warmup_states`）上跑，且在 worker 发起首次 GPU 计算之前，
> 否则 app 会 AV。

---

## 从源码构建（已验证）

### 前置依赖

- **Windows 10/11**
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
  **2025.3** —— 提供 SYCL 编译器（`icx`/`icpx`）和 `sycl8.dll`。
  2025.3 是 **已验证** 的工具链。更新（2026.1，`sycl9.dll`）和更旧版本在本仓库
  未测试。请使用 2025.3。
- [Ninja](https://ninja-build.org/) 1.11+（`winget install Ninja-build.Ninja`）
- Visual Studio 2022（Build Tools 或 IDE）—— 含 "Desktop development with C++"
- CMake 3.22+
- Git（仓库使用 submodule：whisper.cpp、wxWidgets、portaudio、oneDNN）

> SYCL core 的 GPU 内核由 whisper.cpp 的 `icx` + Ninja *子构建*（CMake
> `ExternalProject` 驱动）编译，把 SPIR-V device image 打包进 `ggml-sycl.dll`。
> GUI 应用、wxWidgets、portaudio 和薄薄的 SYCL-core wrapper 留在 MSVC ——
> 下面用 `cl` 覆盖就是为保住这个 MSVC/icx 分工。

### 配置 + 构建

```powershell
# 1. 使用 oneAPI 2025.3（已验证工具链；根 setvars.bat 会加载最新版 —— 见 Notes）。
$env:VS2022INSTALLDIR = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
& "C:\Program Files (x86)\Intel\oneAPI\2025.3\oneapi-vars.bat"

# 2. 配置。顶层强制用 `cl` —— wxWidgets 拒绝 icx（"Unknown WIN32 compiler type"）。
#    whisper.cpp + oneDNN 子构建仍用 icx（它们自己设 -DCMAKE_C_COMPILER=icx）。
#    Ninja 是单配置，Release 在配置时固定。
cmake -S . -B build-gpu -G Ninja `
      -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_C_COMPILER=cl `
      -DCMAKE_CXX_COMPILER=cl

# 3. 构建应用（和无头测试用于验证）。
cmake --build build-gpu --parallel
```

> 顶层的 `-DGGML_SYCL` / `-DGGML_SYCL_DNN` 是 **空操作** —— whisper.cpp 子构建
> 已硬编码它们。别靠顶层开关切换 SYCL。

### 验证

跑无头 pipeline（先 CPU —— 确定性；再 GPU）：

```powershell
.\build-gpu\tests\streaming_pipeline\test_pipeline.exe `
    --model models\ggml-tiny.bin `
    --audio tests\bench_vad\trump_60s_final.wav --cpu
# → ALL PASSED (failures=0)  /  RESULT: GOT TEXT

# GPU 路径（Intel Arc device 0）：
.\build-gpu\tests\streaming_pipeline\test_pipeline.exe `
    --model models\ggml-tiny.bin `
    --audio tests\bench_vad\trump_60s_final.wav --device 0
```

GUI 应用位于 `build-gpu\whisper-xpu-app\Release\whisper_xpu_app.exe`。
要么从已 source 了 `oneapi-vars.bat` 的 shell 启动，要么等 oneAPI 运行时 DLL
已 co-locate 到 exe 旁后双击（构建会自动 copy 大部分 —— 见下面的运行时 DLL 列表）。

### 运行时 DLL

构建会把以下 DLL co-locate 到每个可执行文件旁：

- `whisper_xpu_sycl_core.dll` —— SYCL core（engine、merge_segments、device_detect）
- `whisper.dll`、`ggml.dll`、`ggml-base.dll`、`ggml-cpu.dll` —— CPU/backend 链
- `ggml-sycl.dll`（约 120 MB，icx/Ninja 链接打包了 SPIR-V device image）
- oneAPI 运行时（2025.3 下存在的会 copy）：`sycl8.dll`、`ur_loader.dll`、
  `ur_adapter_level_zero.dll`、`libmmd.dll`、`libiomp5md.dll`、`tcm.dll`、
  oneMKL SYCL + Level Zero adapters 等

未 copy 的 oneAPI DLL 通过 `oneapi-vars.bat` 的 `PATH` 解析。要在不 source oneAPI
的情况下干净运行，从 `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin\`
把缺的 DLL copy 到 exe 旁。

---

## 发布打包

构建一个开箱即用的 zip（见 PR #33）：

```powershell
cmake -S . -B build-gpu -DWHISPER_XPU_RELEASE=ON
cmake --build build-gpu --target release_package
# → build-gpu\whisper_xpu_v1.0.0.zip
```

zip 内含 exe、所有运行时 DLL、`ggml-tiny.bin` + VAD 模型、OpenCC 中文转换词典、
和一个默认 `whisper_xpu.ini`（tiny + CPU）。默认 `OFF` —— 开发构建跳过打包开销。

---

## Notes（注意事项）

编译前值得了解的一些非显然约束。

- **使用 oneAPI 2025.3。** 2025.3（含 `sycl8.dll`）是 GPU 路径已验证的工具链。
  更新（2026.1，`sycl9.dll`）在本仓库未测试 —— 不代表它坏了，只是这里没验证。
  根 `setvars.bat` 会加载最新版 —— 用按版本的 `2025.3\oneapi-vars.bat` 来使用它。
- **`-G Ninja` 下顶层强制 `cl`。** source oneapi-vars 后 CMake 会优先选 PATH 上的
  `icx`；wxWidgets 随即因 `Unknown WIN32 compiler type` 中止。传
  `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl` —— whisper.cpp 和 oneDNN 子构建
  仍用 `icx`（它们自己设）。
- **`ggml-sycl.dll` 必须带 SPIR-V device image。** `lld-link` 静默忽略 `-fsycl`，
  导致 DLL 没有 GPU 内核，首次计算报 `No kernel named im2col_sycl<half>`。
  修复是打包 SPIR-V image 的 `icx`/Ninja 子构建（`icx-cl` 重链也行）。

---

## 项目结构

```
cmake/                      — FindSYCLToolkit.cmake, FindDNNL.cmake
whisper-xpu-core/           — SYCL core DLL (whisper_xpu_sycl_core.dll)
  sycl_src/                   — device detection + SYCL probe sources
whisper-xpu-app/            — wxWidgets GUI app + transcription scheduler
tests/streaming_pipeline/   — headless pipeline unit test (test_pipeline)
third_party/                — whisper.cpp, wxWidgets, portaudio, oneDNN
```

## License

MIT —— 见 [LICENSE](LICENSE)。
