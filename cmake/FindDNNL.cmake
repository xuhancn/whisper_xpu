# FindDNNL.cmake
#
# Auto-detects Intel oneDNN library from the oneAPI installation and delegates
# to the oneAPI-provided dnnl-config.cmake, which defines the DNNL::dnnl
# imported target and handles transitive dependencies (TBB, OpenCL) automatically.
#
# Variables defined:
#   DNNL_FOUND  - True if oneDNN was found via oneAPI
#   DNNLROOT    - Path to the oneDNN installation root
#
# The dnnl-config.cmake from oneAPI provides:
#   DNNL::dnnl  - Imported target with all transitive deps

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
    # Provide the paths from the oneAPI compiler installation so find_package(OpenCL) resolves.
    # These paths are set as hints before loading dnnl-config.cmake.
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(_ocl_root "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
        if(EXISTS "${_ocl_root}/include/CL/opencl.h")
            list(APPEND CMAKE_INCLUDE_PATH "${_ocl_root}/include")
            list(APPEND CMAKE_LIBRARY_PATH "${_ocl_root}/lib")
            message(STATUS "OpenCL path from oneAPI: ${_ocl_root}")
        endif()
    endif()

    # Delegate to the oneAPI-provided dnnl-config.cmake.
    # This defines DNNL::dnnl and handles TBB, OpenCL, etc.
    set(DNNL_DIR "${DNNLROOT}/lib/cmake/dnnl")
    if(EXISTS "${DNNL_DIR}/dnnl-config.cmake")
        include("${DNNL_DIR}/dnnl-config.cmake")
        if(TARGET DNNL::dnnl)
            set(DNNL_FOUND TRUE)
            # Log the oneDNN library path for verification
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
