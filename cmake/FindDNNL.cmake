if(TARGET DNNL::dnnl)
    return()
endif()

option(ONEDNN_STATIC "Link oneDNN as static library (build from source)" ON)

if(ONEDNN_STATIC)
    include(ExternalProject)

    set(ONEDNN_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/oneDNN")
    set(ONEDNN_PREFIX  "${CMAKE_BINARY_DIR}/oneDNN")

    if(NOT EXISTS "${ONEDNN_SRC_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "oneDNN source not found at ${ONEDNN_SRC_DIR}.\n"
            "Run: git submodule update --init third_party/oneDNN")
    endif()

    message(STATUS "oneDNN: building from source (static) at ${ONEDNN_SRC_DIR}")

    include(ProcessorCount)
    ProcessorCount(_proc_cnt)
    if(DEFINED ENV{MAX_JOBS} AND "$ENV{MAX_JOBS}" LESS_EQUAL ${_proc_cnt})
        set(_jobs "$ENV{MAX_JOBS}")
    else()
        set(_jobs "${_proc_cnt}")
    endif()
    unset(_proc_cnt)

    set(_dnnl_build_cmd "${CMAKE_COMMAND}" --build <BINARY_DIR> --config Release --parallel ${_jobs})

    if(WIN32)
        set(DNNL_LIB_NAME "dnnl.lib")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set(_intel_toolset "$ENV{DNNL_INTEL_TOOLSET}")
            message(STATUS "oneDNN: using VS toolset '${_intel_toolset}'")
        else()
            message(FATAL_ERROR "oneDNN: set DNNL_INTEL_TOOLSET in your environment. "
                "Run setvars.bat from oneAPI, then:\n"
                "  set DNNL_INTEL_TOOLSET=Intel C++ Compiler 2025")
        endif()
        ExternalProject_Add(oneDNN_build
            SOURCE_DIR      "${ONEDNN_SRC_DIR}"
            PREFIX          "${ONEDNN_PREFIX}"
            CMAKE_GENERATOR         "${CMAKE_GENERATOR}"
            CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}"
            CMAKE_GENERATOR_TOOLSET  "${_intel_toolset}"
            CMAKE_ARGS
                -DDNNL_GPU_RUNTIME=SYCL
                -DDNNL_CPU_RUNTIME=THREADPOOL
                -DDNNL_BUILD_TESTS=OFF
                -DDNNL_BUILD_EXAMPLES=OFF
                -DONEDNN_BUILD_GRAPH=ON
                -DDNNL_LIBRARY_TYPE=STATIC
                -DDNNL_DPCPP_HOST_COMPILER=DEFAULT
                -DDNNL_ENABLE_ITT_TASKS=OFF
                -DDNNL_ENABLE_JIT_PROFILING=OFF
                -DSYCL_FLAG_SUPPORTED=TRUE
            BUILD_COMMAND ${_dnnl_build_cmd}
            BUILD_BYPRODUCTS "<BINARY_DIR>/src/Release/${DNNL_LIB_NAME}"
            INSTALL_COMMAND ""
        )
    else()
        set(DNNL_LIB_NAME "libdnnl.a")
        ExternalProject_Add(oneDNN_build
            SOURCE_DIR      "${ONEDNN_SRC_DIR}"
            PREFIX          "${ONEDNN_PREFIX}"
            CMAKE_ARGS
                -DDNNL_GPU_RUNTIME=SYCL
                -DDNNL_CPU_RUNTIME=THREADPOOL
                -DDNNL_BUILD_TESTS=OFF
                -DDNNL_BUILD_EXAMPLES=OFF
                -DONEDNN_BUILD_GRAPH=ON
                -DDNNL_LIBRARY_TYPE=STATIC
                -DDNNL_DPCPP_HOST_COMPILER=g++
                -DDNNL_ENABLE_ITT_TASKS=OFF
                -DDNNL_ENABLE_JIT_PROFILING=OFF
                -DSYCL_FLAG_SUPPORTED=TRUE
            CMAKE_CACHE_ARGS
                -DCMAKE_CXX_COMPILER:FILEPATH=icpx
                -DCMAKE_C_COMPILER:FILEPATH=icx
            BUILD_COMMAND ${_dnnl_build_cmd}
            BUILD_BYPRODUCTS "<BINARY_DIR>/src/${DNNL_LIB_NAME}"
            INSTALL_COMMAND ""
        )
    endif()

    ExternalProject_Get_Property(oneDNN_build SOURCE_DIR BINARY_DIR)
    set(ONEDNN_BINARY_DIR "${BINARY_DIR}")
    file(MAKE_DIRECTORY "${ONEDNN_BINARY_DIR}/include")

    if(WIN32)
        set(_lib_dir "${ONEDNN_BINARY_DIR}/src/Release")
    else()
        set(_lib_dir "${ONEDNN_BINARY_DIR}/src")
    endif()
    add_library(DNNL::dnnl STATIC IMPORTED GLOBAL)
    set_target_properties(DNNL::dnnl PROPERTIES
        IMPORTED_LOCATION             "${_lib_dir}/${DNNL_LIB_NAME}"
        IMPORTED_CONFIGURATIONS       "RELEASE"
        IMPORTED_LOCATION_RELEASE     "${_lib_dir}/${DNNL_LIB_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_SRC_DIR}/include;${ONEDNN_BINARY_DIR}/include"
        INTERFACE_LINK_LIBRARIES      "OpenCL.lib;sycl.lib"
    )
    add_dependencies(DNNL::dnnl oneDNN_build)
    set(DNNL_FOUND TRUE)
    message(STATUS "Found oneDNN: static build (DNNL::dnnl)")
    message(STATUS "Found oneDNN: library at ${_lib_dir}/${DNNL_LIB_NAME}")

else()
    set(DNNLROOT "")
    if(DEFINED ENV{DNNLROOT})
        set(DNNLROOT $ENV{DNNLROOT})
        message(STATUS "DNNLROOT from environment: ${DNNLROOT}")
    else()
        message(FATAL_ERROR "oneDNN (dynamic mode) requires DNNLROOT env var. "
            "Run setvars.bat from oneAPI installation.")
    endif()

    if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND DEFINED ENV{CMPLR_ROOT})
        set(_ocl_root "$ENV{CMPLR_ROOT}")
        if(EXISTS "${_ocl_root}/include/CL/opencl.h")
            list(APPEND CMAKE_INCLUDE_PATH "${_ocl_root}/include")
            list(APPEND CMAKE_LIBRARY_PATH "${_ocl_root}/lib")
            message(STATUS "OpenCL path from CMPLR_ROOT: ${_ocl_root}")
        endif()
    endif()

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

set(DNNL_FOUND "${DNNL_FOUND}" CACHE BOOL "oneDNN library found" FORCE)
mark_as_advanced(DNNL_FOUND)
