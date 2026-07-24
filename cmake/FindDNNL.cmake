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

    # ── Parallelism (matches PyTorch) ──
    include(ProcessorCount)
    ProcessorCount(_proc_cnt)
    if(DEFINED ENV{MAX_JOBS} AND "$ENV{MAX_JOBS}" LESS_EQUAL ${_proc_cnt})
        set(_jobs "$ENV{MAX_JOBS}")
    else()
        set(_jobs "${_proc_cnt}")
    endif()
    unset(_proc_cnt)

    # ── Build command (matches PyTorch FindMKLDNN.cmake) ──
    set(_dnnl_build_cmd "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel ${_jobs})
    if(CMAKE_GENERATOR MATCHES "Make|Ninja")
        list(APPEND _dnnl_build_cmd "--" "-l" ${_jobs})
    endif()

    # ── Compiler selection ──
    if(WIN32)
        # With Visual Studio generator, CMAKE_CXX_COMPILER is ignored;
        # the compiler is selected via platform toolset.  PyTorch's
        # FindMKLDNN.cmake sets -DCMAKE_CXX_COMPILER=icx which works on
        # Linux (Ninja/Make) but NOT on Windows (VS).  Dynamically detect
        # the latest installed Intel C++ toolset from the VS installation.
        set(_dnnl_cxx_driver "icx")
        set(_dnnl_host_compiler "DEFAULT")
        set(DNNL_LIB_NAME "dnnl.lib")
        set(_vs_toolset_dir
            "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/MSBuild/Microsoft/VC/v170/Platforms/x64/PlatformToolsets")
        set(_intel_toolset "")
        file(GLOB _toolsets RELATIVE "${_vs_toolset_dir}" "${_vs_toolset_dir}/Intel*")
        list(SORT _toolsets)
        foreach(_ts IN LISTS _toolsets)
            if(_ts MATCHES "Intel C\\+\\+ Compiler")
                set(_intel_toolset "${_ts}")
            endif()
        endforeach()
        if(_intel_toolset)
            message(STATUS "oneDNN: using VS toolset '${_intel_toolset}'")
        else()
            message(FATAL_ERROR "oneDNN: Intel C++ toolset not found. "
                "Run setvars.bat from oneAPI installation.")
        endif()
        set(_dnnl_toolset_arg
            CMAKE_GENERATOR         "${CMAKE_GENERATOR}"
            CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}"
            CMAKE_GENERATOR_TOOLSET  "${_intel_toolset}"
        )
    else()
        set(_dnnl_cxx_driver "icpx")
        set(_dnnl_host_compiler "g++")
        set(DNNL_LIB_NAME "libdnnl.a")
        set(_dnnl_toolset_arg
            CMAKE_CXX_COMPILER "icpx"
            CMAKE_C_COMPILER   "icpx"
        )
    endif()

    # ── oneDNN build (PyTorch FindMKLDNN.cmake pattern) ──
    ExternalProject_Add(oneDNN_build
        SOURCE_DIR      "${ONEDNN_SRC_DIR}"
        PREFIX          "${ONEDNN_PREFIX}"
        ${_dnnl_toolset_arg}
        CMAKE_ARGS
            -DCMAKE_CXX_COMPILER=${_dnnl_cxx_driver}
            -DDNNL_GPU_RUNTIME=SYCL
            -DDNNL_CPU_RUNTIME=THREADPOOL
            -DDNNL_BUILD_TESTS=OFF
            -DDNNL_BUILD_EXAMPLES=OFF
            -DONEDNN_BUILD_GRAPH=ON
            -DDNNL_LIBRARY_TYPE=STATIC
            -DDNNL_DPCPP_HOST_COMPILER=${_dnnl_host_compiler}
            # Skip the -fsycl flag check — icx-cl supports it but cmake's
            # try_compile fails under VS generator + Intel toolset.
            -DSYCL_FLAG_SUPPORTED=TRUE
        BUILD_COMMAND ${_dnnl_build_cmd}
        BUILD_BYPRODUCTS "<BINARY_DIR>/src/${DNNL_LIB_NAME}"
        INSTALL_COMMAND ""
    )

    ExternalProject_Get_Property(oneDNN_build SOURCE_DIR BINARY_DIR)
    set(ONEDNN_BINARY_DIR "${BINARY_DIR}")

    # Pre-create the binary include dir so CMake's import-target validation
    # passes at configure time. The ExternalProject populates it during build.
    file(MAKE_DIRECTORY "${ONEDNN_BINARY_DIR}/include")

    # ── Import the static library (matches PyTorch) ──
    # PyTorch: ${BINARY_DIR}/src/dnnl.lib (no config subdir)
    # Includes: ${SOURCE_DIR}/include ${BINARY_DIR}/include
    add_library(DNNL::dnnl STATIC IMPORTED GLOBAL)
    set_target_properties(DNNL::dnnl PROPERTIES
        IMPORTED_LOCATION             "${ONEDNN_BINARY_DIR}/src/${DNNL_LIB_NAME}"
        IMPORTED_CONFIGURATIONS       "RELEASE"
        IMPORTED_LOCATION_RELEASE     "${ONEDNN_BINARY_DIR}/src/${DNNL_LIB_NAME}"
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
