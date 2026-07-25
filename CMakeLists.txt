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

include(cmake/FindSYCLToolkit.cmake)

option(WHISPER_XPU_USE_GPU "Enable GPU acceleration via SYCL" ${SYCL_FOUND})

if(WHISPER_XPU_USE_GPU AND SYCL_FOUND)
    set(HAVE_SYCL TRUE)
    add_compile_definitions(WHISPER_XPU_HAS_SYCL=1)
    message(STATUS "Intel oneAPI SYCL toolkit detected. GPU acceleration enabled.")

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

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(HAVE_SYCL)
    set(GGML_SYCL ON CACHE BOOL "Enable SYCL in whisper.cpp")
    set(GGML_SYCL_DNN ON CACHE BOOL "Enable oneDNN in whisper.cpp SYCL backend")
endif()
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(WHISPER_NO_EXAMPLES ON CACHE BOOL "" FORCE)
set(WHISPER_NO_TESTS ON CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "Build ggml as static library" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
add_subdirectory(third_party/whisper.cpp)

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

# ── Install rules ──
include(GNUInstallDirs)

# Release package: include/ + lib/ + sycl.dll
set_target_properties(whisper_xpu_core PROPERTIES
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/whisper-xpu-core/whisper_xpu_core.h"
)
install(TARGETS whisper_xpu_core
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# SYCL runtime DLL (required at runtime)
if(HAVE_SYCL AND SYCL_LIBRARY_DIR)
    install(FILES ${SYCL_LIBRARY_DIR}/sycl.dll DESTINATION ${CMAKE_INSTALL_BINDIR}
        OPTIONAL)
endif()
