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
    #  Approach matches PyTorch's FindMKLDNN.cmake:
    #  https://github.com/pytorch/pytorch/blob/0cfa492631cca99c2aa4161ec67035fcda976869/cmake/Modules/FindMKLDNN.cmake
    # ═══════════════════════════════════════════════════════════════
    include(ExternalProject)

    set(ONEDNN_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/oneDNN")
    set(ONEDNN_PREFIX  "${CMAKE_BINARY_DIR}/oneDNN")

    if(NOT EXISTS "${ONEDNN_SRC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "oneDNN source not found at ${ONEDNN_SRC_DIR}.\n"
            "Run: git submodule update --init third_party/oneDNN")
    endif()

    message(STATUS "oneDNN: building from source (static) at ${ONEDNN_SRC_DIR}")

    # ── Parallelism ──
    include(ProcessorCount)
    ProcessorCount(_proc_cnt)
    if(DEFINED ENV{MAX_JOBS} AND "$ENV{MAX_JOBS}" LESS_EQUAL ${_proc_cnt})
        set(_jobs "$ENV{MAX_JOBS}")
    else()
        set(_jobs "${_proc_cnt}")
    endif()
    unset(_proc_cnt)

    # ── Build-system arguments ──
    if(WIN32)
        get_property(_dnnl_host_cxx CACHE CMAKE_CXX_COMPILER PROPERTY VALUE)
        if("${_dnnl_host_cxx}" STREQUAL "")
            set(_dnnl_host_cxx "cl")
        endif()
        set(_dnnl_cmake_gen
            CMAKE_GENERATOR          "${CMAKE_GENERATOR}"
            CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}"
        )
        set(_dnnl_cxx_flags "/MP")
        set(_dnnl_build_cmd "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --parallel ${_jobs})
    else()
        set(_dnnl_cmake_gen
            CMAKE_CXX_COMPILER "icpx"
            CMAKE_C_COMPILER   "icpx"
        )
        set(_dnnl_cxx_flags "")
        set(_dnnl_build_cmd "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release -j ${_jobs})
    endif()

    # ── oneDNN build configuration (matches PyTorch) ──
    # DNNL_CPU_RUNTIME=NONE  → no CPU backend
    # DNNL_GPU_RUNTIME=SYCL  → GPU via SYCL
    # INSTALL_COMMAND ""     → build in-place, reference BINARY_DIR
    ExternalProject_Add(oneDNN_build
        SOURCE_DIR        "${ONEDNN_SRC_DIR}"
        PREFIX            "${ONEDNN_PREFIX}"
        ${_dnnl_cmake_gen}
        CMAKE_ARGS
            -DCMAKE_CXX_COMPILER=icx
            -DCMAKE_C_COMPILER=icx
            -DNNL_LIBRARY_TYPE=STATIC
            -DNNL_GPU_RUNTIME=SYCL
            -DNNL_CPU_RUNTIME=NONE
            -DNNL_BUILD_TESTS=OFF
            -DNNL_BUILD_EXAMPLES=OFF
            -DNNL_ENABLE_CONCURRENT_EXEC=ON
            -DNNL_EXPERIMENTAL=ON
            -DONEDNN_BUILD_GRAPH=ON
            -DNNL_DPCPP_HOST_COMPILER=${_dnnl_host_cxx}
            -DCMAKE_CXX_FLAGS=${_dnnl_cxx_flags}
        BUILD_COMMAND ${_dnnl_build_cmd}
        INSTALL_COMMAND ""
    )

    # ── Determine library name ──
    if(WIN32)
        set(DNNL_LIB_NAME "dnnl.lib")
    else()
        set(DNNL_LIB_NAME "libdnnl.a")
    endif()

    ExternalProject_Get_Property(oneDNN_build SOURCE_DIR BINARY_DIR)
    set(ONEDNN_BINARY_DIR "${BINARY_DIR}")

    # Pre-create the binary include dir so CMake's import-target validation
    # passes at configure time. The ExternalProject populates it during build.
    file(MAKE_DIRECTORY "${ONEDNN_BINARY_DIR}/include")

    # ── Import the static library as DNNL::dnnl ──
    # Include paths: source headers + build-generated (dnnl_config.h).
    # Library: from build tree (INSTALL_COMMAND is empty).
    # Note: VS multi-config generator puts output in Release/ subdir.
    add_library(DNNL::dnnl STATIC IMPORTED GLOBAL)
    set_target_properties(DNNL::dnnl PROPERTIES
        IMPORTED_LOCATION             "${ONEDNN_BINARY_DIR}/src/Release/${DNNL_LIB_NAME}"
        IMPORTED_CONFIGURATIONS       "RELEASE"
        IMPORTED_LOCATION_RELEASE     "${ONEDNN_BINARY_DIR}/src/Release/${DNNL_LIB_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_SRC_DIR}/include;${ONEDNN_BINARY_DIR}/include"
        INTERFACE_LINK_LIBRARIES      "OpenCL.lib;sycl.lib"
    )
    add_dependencies(DNNL::dnnl oneDNN_build)
    set(DNNL_FOUND TRUE)
    message(STATUS "Found oneDNN: static build (DNNL::dnnl)")
    message(STATUS "Found oneDNN: library at ${ONEDNN_BINARY_DIR}/src/${DNNL_LIB_NAME}")

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
