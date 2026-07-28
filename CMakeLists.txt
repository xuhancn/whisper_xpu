cmake_minimum_required(VERSION 3.22)
project(whisper_xpu VERSION 0.1.0 LANGUAGES C CXX)

if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/MP>)

# ── SYCL is mandatory ──
include(cmake/FindSYCLToolkit.cmake)
if(NOT SYCL_FOUND)
    message(FATAL_ERROR
        "Intel oneAPI SYCL toolkit not found.\n"
        "This project requires SYCL for XPU/GPU acceleration.\n"
        "Install the Intel oneAPI Base Toolkit:\n"
        "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html\n"
        "After installation, run setvars.bat (Windows) or source setvars.sh (Linux).")
endif()

set(HAVE_SYCL TRUE)
add_compile_definitions(WHISPER_XPU_HAS_SYCL=1)
message(STATUS "Intel oneAPI SYCL toolkit detected. GPU acceleration enabled.")

# ── oneDNN detection ──
option(ONEDNN_STATIC "Link oneDNN as static library (build from source)" ON)
include(cmake/FindDNNL.cmake)

# ── whisper.cpp ──
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(GGML_SYCL ON CACHE BOOL "Enable SYCL in whisper.cpp")
set(GGML_SYCL_DNN ON CACHE BOOL "Enable oneDNN in whisper.cpp SYCL backend")
set(MKL_LINK "dynamic" CACHE STRING "MKL link mode (oneAPI 2026.1: dynamic; static SYCL libs no longer shipped)" FORCE)
set(MKL_SYCL_LINK "dynamic" CACHE STRING "MKL SYCL link mode (oneAPI 2026.1: only 'dynamic' supported)" FORCE)
set(MKL_SYCL_THREADING "intel_thread" CACHE STRING "MKL SYCL threading: use Intel OpenMP instead of TBB")
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(WHISPER_NO_EXAMPLES ON CACHE BOOL "" FORCE)
set(WHISPER_NO_TESTS ON CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "Build ggml as static library" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
add_subdirectory(third_party/whisper.cpp)

# Remove ggml-sycl from ggml's transitive link — the SYCL backend lives
# in whisper_xpu_sycl_core.dll exclusively.  The CPU DLL must not
# statically link ggml-sycl or it conflicts with sycl_core imports.
get_target_property(_ggml_iface ggml INTERFACE_LINK_LIBRARIES)
if(_ggml_iface)
    message(STATUS "ggml INTERFACE_LINK_LIBRARIES before filter: ${_ggml_iface}")
    list(FILTER _ggml_iface EXCLUDE REGEX "ggml-sycl")
    message(STATUS "ggml INTERFACE_LINK_LIBRARIES after filter:  ${_ggml_iface}")
    set_target_properties(ggml PROPERTIES INTERFACE_LINK_LIBRARIES "${_ggml_iface}")
endif()

set(wxBUILD_SHARED OFF CACHE BOOL "" FORCE)
set(wxUSE_STC OFF CACHE BOOL "" FORCE)
set(wxUSE_WEBVIEW OFF CACHE BOOL "" FORCE)
set(wxUSE_MEDIACTRL OFF CACHE BOOL "" FORCE)
set(wxUSE_PRINTING_ARCH OFF CACHE BOOL "" FORCE)
set(wxUSE_HTML ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/wxWidgets)

set(PA_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/portaudio)

add_subdirectory(whisper-xpu-core)
add_subdirectory(whisper-xpu-app)
add_subdirectory(tests)

# ── Release package ──
configure_file(cmake/release.cmake.in "${CMAKE_BINARY_DIR}/release.cmake" @ONLY)
add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/release.cmake"
    DEPENDS whisper_xpu_app
    COMMENT "Creating release package in ${CMAKE_BINARY_DIR}/release/")
