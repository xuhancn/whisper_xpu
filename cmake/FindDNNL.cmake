# FindDNNL.cmake
#
# Locates Intel oneDNN library. Supports two modes controlled by ONEDNN_STATIC:
#
#   ONEDNN_STATIC=ON  (default)
#     - Builds oneDNN from source (third_party/oneDNN) as a static library.
#     - Uses Intel C++ Compiler (icx/icpx) so SYCL GPU runtime works.
#     - Produces DNNL::dnnl as a STATIC IMPORTED target.
#     - Does NOT depend on the oneAPI pre-built package (no dnnl.dll needed).
#
#   ONEDNN_STATIC=OFF
#     - Uses pre-built oneDNN from the oneAPI installation (dnnl.dll).
#     - Auto-detects DNNLROOT by scanning oneAPI install paths or $ENV{DNNLROOT}.
#     - Delegates to oneAPI-provided dnnl-config.cmake (defines DNNL::dnnl SHARED).
#
# Variables defined:
#   DNNL_FOUND          - True if oneDNN is available
#   DNNLROOT            - Root directory of the oneDNN installation

# ── Guard against double processing ──
# This module is both included from the root CMakeLists.txt and found
# via find_package(DNNL) inside ggml-sycl's cmake. Skip if the target
# already exists (within a single configure run).
# NOTE: we check TARGET not DNNL_FOUND because the cache variable
# persists across re-configures but targets do not.
if(TARGET DNNL::dnnl)
    return()
endif()

# ── Option: static or dynamic linking ──
option(ONEDNN_STATIC "Link oneDNN as static library (build from source)" ON)

# ── Record the CMAKE_MODULE_PATH as it was when this file first runs ──
# The parent scope (root CMakeLists.txt) PREPENDs our cmake/ dir to
# CMAKE_MODULE_PATH. Capture it so ggml-sycl's find_package(DNNL) still
# resolves to this same module.
if(ONEDNN_STATIC)
    # ═══════════════════════════════════════════════════════════════
    #  MODE 1: Static linking — build oneDNN from source
    # ═══════════════════════════════════════════════════════════════
    include(ExternalProject)

    set(ONEDNN_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/oneDNN")
    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/oneDNN_install")

    if(NOT EXISTS "${ONEDNN_SRC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "oneDNN source not found at ${ONEDNN_SRC_DIR}.\n"
            "Run: git submodule update --init third_party/oneDNN")
    endif()

    message(STATUS "oneDNN: building from source (static) at ${ONEDNN_SRC_DIR}")

    # ── Build-system arguments ──
    # On Windows with Visual Studio, use the VS generator with the Intel C++
    # Compiler toolset so SYCL GPU code compiles.  On Linux / Ninja, pass
    # CMAKE_CXX_COMPILER=icpx directly.
    if(WIN32)
        if(CMAKE_GENERATOR MATCHES "Visual Studio")
            set(_dnnl_cmake_gen
                CMAKE_GENERATOR             "${CMAKE_GENERATOR}"
                CMAKE_GENERATOR_PLATFORM    "${CMAKE_GENERATOR_PLATFORM}"
                CMAKE_GENERATOR_TOOLSET     "Intel C++ Compiler 2025"
            )
        else()
            set(_dnnl_cmake_gen
                CMAKE_GENERATOR         "Ninja"
                CMAKE_CXX_COMPILER      "icx"
            )
        endif()
    else()
        set(_dnnl_cmake_gen
            CMAKE_CXX_COMPILER "icpx"
        )
    endif()

    # ── oneDNN build configuration ──
    # DNNL_CPU_RUNTIME=THREADPOOL avoids pulling in TBB.
    # DNNL_GPU_RUNTIME=SYCL enables SYCL GPU acceleration.
    #
    # --parallel is safe here because /MP is only applied to MSVC targets
    # (via generator expression), so icx-cl (Intel compiler) for oneDNN
    # won't receive conflicting flags.
    ExternalProject_Add(oneDNN_build
        SOURCE_DIR        "${ONEDNN_SRC_DIR}"
        PREFIX            "${CMAKE_BINARY_DIR}/oneDNN"
        INSTALL_DIR       "${ONEDNN_INSTALL_DIR}"
        ${_dnnl_cmake_gen}
        CMAKE_ARGS
            -DNNL_LIBRARY_TYPE=STATIC
            -DNNL_GPU_RUNTIME=SYCL
            -DNNL_CPU_RUNTIME=THREADPOOL
            -DNNL_BUILD_TESTS=OFF
            -DNNL_BUILD_EXAMPLES=OFF
            -DNNL_ENABLE_CONCURRENT_EXEC=ON
            -DNNL_EXPERIMENTAL=ON
            -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
            -DCMAKE_CXX_FLAGS=/MP
        BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --parallel
        BUILD_BYPRODUCTS
            "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib"
    )

    # ── Import the static library as DNNL::dnnl ──
    # Set both generic IMPORTED_LOCATION and per-config locations
    # because ggml-sycl's cmake iterates IMPORTED_CONFIGURATIONS
    # and reads each per-config location.
    add_library(DNNL::dnnl STATIC IMPORTED GLOBAL)
    set_target_properties(DNNL::dnnl PROPERTIES
        IMPORTED_LOCATION             "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib"
        IMPORTED_CONFIGURATIONS       "RELEASE"
        IMPORTED_LOCATION_RELEASE     "${ONEDNN_INSTALL_DIR}/lib/dnnl.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_SRC_DIR}/include"
        INTERFACE_LINK_LIBRARIES      "OpenCL.lib;sycl.lib"
    )
    add_dependencies(DNNL::dnnl oneDNN_build)

    set(DNNL_FOUND TRUE)
    set(DNNLROOT "${ONEDNN_INSTALL_DIR}")
    message(STATUS "Found oneDNN: static build (DNNL::dnnl)")
    message(STATUS "Found oneDNN: library at ${ONEDNN_INSTALL_DIR}/lib/dnnl.lib")

else()
    # ═══════════════════════════════════════════════════════════════
    #  MODE 2: Dynamic linking — use oneAPI pre-built DLL
    # ═══════════════════════════════════════════════════════════════

    # -- Determine DNNLROOT --
    set(DNNLROOT "")
    if(DEFINED ENV{DNNLROOT})
        set(DNNLROOT $ENV{DNNLROOT})
        message(STATUS "DNNLROOT from environment: ${DNNLROOT}")
    else()
        # Scan common oneAPI installation paths on Windows
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            foreach(_version "latest" "2025.3" "2025.2" "2025.1" "2024.2" "2024.1")
                set(_candidate "C:/Program Files (x86)/Intel/oneAPI/dnnl/${_version}")
                if(EXISTS "${_candidate}/lib/cmake/dnnl/dnnl-config.cmake")
                    set(DNNLROOT "${_candidate}")
                    message(STATUS "Found oneDNN at: ${DNNLROOT}")
                    break()
                endif()
            endforeach()
        elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
            foreach(_version "latest" "2025.3" "2025.2" "2025.1" "2024.2" "2024.1")
                set(_candidate "/opt/intel/oneapi/dnnl/${_version}")
                if(EXISTS "${_candidate}/lib/cmake/dnnl/dnnl-config.cmake")
                    set(DNNLROOT "${_candidate}")
                    message(STATUS "Found oneDNN at: ${DNNLROOT}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    string(COMPARE EQUAL "${DNNLROOT}" "" nodnnl)
    if(nodnnl)
        set(DNNL_FOUND FALSE)
        message(STATUS "oneDNN not found. SYCL backend will build without oneDNN acceleration.")
    else()
        # The oneAPI dnnl-config.cmake requires OpenCL headers/libs (for SYCL runtime).
        # Provide paths from the oneAPI compiler installation.
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            set(_ocl_root "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
            if(EXISTS "${_ocl_root}/include/CL/opencl.h")
                list(APPEND CMAKE_INCLUDE_PATH "${_ocl_root}/include")
                list(APPEND CMAKE_LIBRARY_PATH "${_ocl_root}/lib")
                message(STATUS "OpenCL path from oneAPI: ${_ocl_root}")
            endif()
        endif()

        # Delegate to the oneAPI-provided dnnl-config.cmake.
        set(DNNL_DIR "${DNNLROOT}/lib/cmake/dnnl")
        if(EXISTS "${DNNL_DIR}/dnnl-config.cmake")
            include("${DNNL_DIR}/dnnl-config.cmake")
            if(TARGET DNNL::dnnl)
                set(DNNL_FOUND TRUE)
                get_target_property(_configs DNNL::dnnl IMPORTED_CONFIGURATIONS)
                if(_configs)
                    foreach(_cfg ${_configs})
                        get_target_property(_lib DNNL::dnnl IMPORTED_LOCATION_${_cfg})
                        if(_lib)
                            message(STATUS "Found oneDNN: ${_lib}")
                            break()
                        endif()
                    endforeach()
                endif()
            else()
                set(DNNL_FOUND FALSE)
                message(STATUS "oneDNN config found but DNNL::dnnl target not defined")
            endif()
        else()
            set(DNNL_FOUND FALSE)
            message(STATUS "dnnl-config.cmake not found in ${DNNL_DIR}")
        endif()
    endif()
endif()

# ── Guard against double-processing by find_package(DNNL) later ──
# Export the DNNL_FOUND decision into the cache so that subsequent
# find_package(DNNL) calls from ggml-sycl see it.
set(DNNL_FOUND "${DNNL_FOUND}" CACHE BOOL "oneDNN library found" FORCE)
mark_as_advanced(DNNL_FOUND)
