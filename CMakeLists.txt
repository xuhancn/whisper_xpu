cmake_minimum_required(VERSION 3.22)
project(whisper_xpu VERSION 0.1.0 LANGUAGES C CXX)

# Allow submodules that use cmake_minimum_required(VERSION < 3.5)
# to work with CMake 4.x (e.g. wxWidgets 3.2.6).
if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Multi-process compilation for MSVC targets only.
# /MP compiles multiple source files in parallel within a single project.
# Intel C++ targets (SYCL) are excluded — icx-cl rejects /MP with -fsycl.
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>)

# ── SYCL toolchain detection ──
# FindSYCLToolkit locates the Intel oneAPI installation (compiler, headers, libs).
include(cmake/FindSYCLToolkit.cmake)

# Option to enable GPU acceleration. Default to ON when SYCL toolkit is found.
option(WHISPER_XPU_USE_GPU "Enable GPU acceleration via SYCL" ${SYCL_FOUND})

# When the SYCL toolkit is available and GPU is requested, set HAVE_SYCL.
# On Windows with Visual Studio, this means compiling SYCL-specific files
# with the Intel C++ Compiler toolset, while other files use MSVC.
# whisper.cpp/ggml's SYCL backend handles this automatically.
if(WHISPER_XPU_USE_GPU AND SYCL_FOUND)
    set(HAVE_SYCL TRUE)
    add_compile_definitions(WHISPER_XPU_HAS_SYCL=1)
    message(STATUS "Intel oneAPI SYCL toolkit detected. GPU acceleration enabled.")

    # ── OneDNN detection (for SYCL backend acceleration) ──
    # ONEDNN_STATIC controls linking mode:
    #   ON  (default) — build oneDNN from source as static lib (no dnnl.dll needed)
    #   OFF           — use pre-built oneDNN DLL from oneAPI installation
    option(ONEDNN_STATIC "Link oneDNN as static library (build from source)" ON)
    include(cmake/FindDNNL.cmake)
else()
    set(HAVE_SYCL FALSE)
    set(DNNL_FOUND FALSE)
    if(NOT SYCL_FOUND)
        message(STATUS "Intel oneAPI SYCL toolkit not found. Building CPU-only. "
                       "Install oneAPI Base Toolkit for GPU support.")
    else()
        message(STATUS "GPU acceleration disabled by WHISPER_XPU_USE_GPU=OFF")
    endif()
endif()

# ── whisper.cpp ──
# Add our cmake/ dir to CMAKE_MODULE_PATH so FindDNNL.cmake resolves
# when ggml's SYCL backend calls find_package(DNNL).
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(HAVE_SYCL)
    set(GGML_SYCL ON CACHE BOOL "Enable SYCL in whisper.cpp")
    # oneDNN integration in ggml's SYCL backend.
    set(GGML_SYCL_DNN ON CACHE BOOL "Enable oneDNN in whisper.cpp SYCL backend")
endif()
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(WHISPER_NO_EXAMPLES ON CACHE BOOL "" FORCE)
set(WHISPER_NO_TESTS ON CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "Build ggml as static library" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
add_subdirectory(third_party/whisper.cpp)

# ── wxWidgets (static, minimal) ──
set(wxBUILD_SHARED OFF CACHE BOOL "" FORCE)
set(wxUSE_STC OFF CACHE BOOL "" FORCE)
set(wxUSE_WEBVIEW OFF CACHE BOOL "" FORCE)
set(wxUSE_MEDIACTRL OFF CACHE BOOL "" FORCE)
set(wxUSE_PRINTING_ARCH OFF CACHE BOOL "" FORCE)
set(wxUSE_HTML ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/wxWidgets)

# ── portaudio (static only) ──
set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/portaudio)

# ── Own targets ──
add_subdirectory(whisper-xpu-core)
add_subdirectory(whisper-xpu-app)
