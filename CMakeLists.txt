cmake_minimum_required(VERSION 3.22)
project(whisper_xpu VERSION 0.1.0 LANGUAGES C CXX)

if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(ExternalProject)

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

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── whisper.cpp: built via Ninja + icx ExternalProject ──────────────────
# Under the Visual Studio generator ggml-sycl's LINK step is driven by
# lld-link (not xilink), which IGNORES -fsycl → sycl-post-link never runs →
# no SPIR-V device image is bundled into the DLL → runtime
# "No kernel named im2col_sycl<half>".  Under Ninja, icx IS the link driver:
# it honors -fsycl and runs sycl-post-link + clang-offload-bundler, bundling
# the device image INTO ggml-sycl.dll.  So the whole whisper.cpp tree
# (ggml, ggml-base, ggml-cpu, ggml-sycl, whisper — plus oneDNN via FindDNNL)
# is built in a Ninja+icx sub-build, exactly the oneDNN pattern used in
# cmake/FindDNNL.cmake.  The MSVC-driven main build (core + app + wxWidgets +
# portaudio + gtest) then links the resulting SHARED DLLs as import libs.
# Only the SYCL-bearing libraries use Ninja; our own code stays MSVC.
set(_whisper_src "${CMAKE_SOURCE_DIR}/third_party/whisper.cpp")
set(_whisper_bld "${CMAKE_BINARY_DIR}/whisper_cpp/src/whisper_cpp_build-build")
set(_whisper_lib "${CMAKE_BINARY_DIR}/whisper_cpp/lib")
# whisper.cpp's top-level CMakeLists force-sets CMAKE_RUNTIME_OUTPUT_DIRECTORY
# and CMAKE_LIBRARY_OUTPUT_DIRECTORY to ${CMAKE_BINARY_DIR}/bin as NORMAL vars
# (not CACHE), which shadow any -D cache value passed in — so the sub-build's
# DLLs always land in <sub-build>/bin regardless of output-dir args.  Point
# _whisper_bin there directly.  whisper.cpp does NOT override
# CMAKE_ARCHIVE_OUTPUT_DIRECTORY, so the -D below is honored and the .lib
# import libs land in _whisper_lib (where test_pipeline links them from).
set(_whisper_bin "${_whisper_bld}/bin")
# The sub-build runs icx as BOTH compiler and linker.  MSBuild's ExternalProject
# custom-build step must see the oneAPI environment (LIB/INCLUDE/PATH/CMPLR_ROOT
# ...) so icx's link test can find libircmt.lib, etc.  We capture the live env
# HERE (main configure runs under oneapi-vars) into a batch wrapper and `call`
# it before each sub-build cmake invocation.
#
# NOTE: passing the env via `cmake -E env "LIB=$ENV{LIB}" ...` does NOT work on
# Windows.  LIB/INCLUDE/PATH are ';'-separated path lists, and ExternalProject
# stores COMMAND as a ';'-joined CMake list — so each path component becomes a
# SEPARATE argv token (only the first keeps its `LIB=` prefix).  cmake -E env
# then hits a bare path component before `--`, treats it as the command to exec
# (a directory), and dies with "no such file or directory".  Serializing to a
# bat sidesteps the splitting entirely: semicolons are literal inside
# `set "VAR=a;b;c"`.
set(_env_bat "${CMAKE_BINARY_DIR}/whisper_cpp_env.bat")
file(TO_NATIVE_PATH "${_env_bat}" _env_bat_n)
file(WRITE "${_env_bat}"
    "@echo off\n"
    "set \"LIB=$ENV{LIB}\"\n"
    "set \"INCLUDE=$ENV{INCLUDE}\"\n"
    "set \"PATH=$ENV{PATH}\"\n"
    "set \"CMPLR_ROOT=$ENV{CMPLR_ROOT}\"\n"
    "set \"ONEAPI_ROOT=$ENV{ONEAPI_ROOT}\"\n"
    "set \"MKLROOT=$ENV{MKLROOT}\"\n")
ExternalProject_Add(whisper_cpp_build
    SOURCE_DIR        "${_whisper_src}"
    PREFIX            "${CMAKE_BINARY_DIR}/whisper_cpp"
    CONFIGURE_COMMAND call "${_env_bat_n}" &&
                      ${CMAKE_COMMAND} -G Ninja
                      -S "${_whisper_src}" -B "${_whisper_bld}"
                      -DCMAKE_C_COMPILER=icx
                      -DCMAKE_CXX_COMPILER=icx
                      -DCMAKE_BUILD_TYPE=Release
                      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${_whisper_lib}
                      -DCMAKE_MODULE_PATH=${CMAKE_SOURCE_DIR}/cmake
                      -DONEDNN_STATIC=ON
                      -DONEDNN_SRC_DIR=${CMAKE_SOURCE_DIR}/third_party/oneDNN
                      -DGGML_SYCL=ON
                      -DGGML_SYCL_DNN=ON
                      -DMKL_LINK=dynamic
                      -DMKL_SYCL_LINK=dynamic
                      -DMKL_SYCL_THREADING=intel_thread
                      -DGGML_CUDA=OFF
                      -DGGML_VULKAN=OFF
                      -DWHISPER_NO_EXAMPLES=ON
                      -DWHISPER_NO_TESTS=ON
                      -DBUILD_SHARED_LIBS=ON
                      -DGGML_STATIC=OFF
    BUILD_COMMAND     call "${_env_bat_n}" &&
                      ${CMAKE_COMMAND} --build "${_whisper_bld}" --parallel
    BUILD_BYPRODUCTS
        "${_whisper_bin}/ggml-base.dll"
        "${_whisper_bin}/ggml-cpu.dll"
        "${_whisper_bin}/ggml-sycl.dll"
        "${_whisper_bin}/ggml.dll"
        "${_whisper_bin}/whisper.dll"
        "${_whisper_lib}/ggml-base.lib"
        "${_whisper_lib}/ggml-cpu.lib"
        "${_whisper_lib}/ggml-sycl.lib"
        "${_whisper_lib}/ggml.lib"
        "${_whisper_lib}/whisper.lib"
    INSTALL_COMMAND ""
)

# IMPORTED targets wrapping the Ninja-built shared libs.  Consumers (core, app,
# tests) link these; whisper_xpu_copy_shared_libs() co-locates the DLLs beside
# each exe at runtime.  ggml pulls its backends (ggml-base, ggml-cpu, ggml-sycl)
# transitively so loading whisper→ggml brings the whole chain up.
set(_wp_inc "${_whisper_src}/include;${_whisper_src}/ggml/include")
function(_wp_import _name)
    add_library(${_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${_name} PROPERTIES
        IMPORTED_LOCATION             "${_whisper_bin}/${_name}.dll"
        IMPORTED_IMPLIB               "${_whisper_lib}/${_name}.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${_wp_inc}")
    add_dependencies(${_name} whisper_cpp_build)
endfunction()
_wp_import(ggml-base)
_wp_import(ggml-cpu)
_wp_import(ggml-sycl)
_wp_import(ggml)
_wp_import(whisper)
set_target_properties(ggml PROPERTIES
    INTERFACE_LINK_LIBRARIES "ggml-base;ggml-cpu;ggml-sycl")
set_target_properties(whisper PROPERTIES
    INTERFACE_LINK_LIBRARIES "ggml")

# ── Co-locate whisper.cpp's shared DLLs beside each consumer exe ──
# whisper_xpu_sycl_core.dll links ggml/whisper/ggml-sycl as IMPORT libraries,
# so at runtime an exe needs ggml.dll + ggml-base.dll + whisper.dll +
# ggml-sycl.dll beside it (they are not on PATH).  The oneAPI runtime DLLs are
# covered by setvars/PATH for tests; the packaged app copies them separately.
function(whisper_xpu_copy_shared_libs _target)
    if(WIN32)
        foreach(_dep ggml-base ggml-cpu ggml whisper ggml-sycl)
            if(TARGET ${_dep})
                add_custom_command(TARGET ${_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${_dep}>
                        $<TARGET_FILE_DIR:${_target}>
                    COMMENT "Copying ${_dep} to ${_target} output directory")
            endif()
        endforeach()
    endif()
endfunction()

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
