# FindSYCLToolkit.cmake
#
# Locates Intel oneAPI SYCL runtime libraries.
# Adapted from: https://github.com/pytorch/pytorch/blob/main/cmake/Modules/FindSYCLToolkit.cmake
#
# Variables defined:
#   SYCL_FOUND              - True if SYCL runtime is found
#   SYCL_COMPILER           - Path to icx.exe
#   SYCL_INCLUDE_DIR        - sycl/sycl.hpp include directory
#   SYCL_LIBRARY_DIR        - sycl library directory
#   SYCL_LIBRARY            - Full path to sycl library
#   SYCL_COMPILER_VERSION   - Numeric version (e.g. 20250101)

include(FindPackageHandleStandardArgs)

# -- Determine SYCL root --
set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
    set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
    set(SYCL_ROOT $ENV{CMPLR_ROOT})
else()
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(SYCL_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(SYCL_ROOT "/opt/intel/oneapi/compiler/latest")
    endif()
    if(NOT EXISTS "${SYCL_ROOT}")
        set(SYCL_ROOT "")
    endif()
endif()

string(COMPARE EQUAL "${SYCL_ROOT}" "" nosyclfound)
if(nosyclfound)
    set(SYCL_FOUND FALSE)
    set(SYCL_REASON_FAILURE "SYCL root not found. Set SYCL_ROOT or CMPLR_ROOT env var.")
    find_package_handle_standard_args(SYCL DEFAULT_MSG SYCL_FOUND)
    return()
endif()

message(STATUS "SYCL root: ${SYCL_ROOT}")

# -- Find SYCL compiler (icx) --
find_program(SYCL_COMPILER
    NAMES icx
    PATHS "${SYCL_ROOT}"
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH
)

if(NOT SYCL_COMPILER)
    set(SYCL_FOUND FALSE)
    set(SYCL_REASON_FAILURE "icx not found in ${SYCL_ROOT}/bin")
    find_package_handle_standard_args(SYCL DEFAULT_MSG SYCL_FOUND)
    return()
endif()

# -- Parse compiler version from icx --version --
function(parse_icx_version version_number)
    execute_process(COMMAND ${SYCL_COMPILER} --version
        OUTPUT_VARIABLE SYCL_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE
        "Intel\\(R\\) (.*) Compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\2"
        SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
    string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
    list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
    list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
    list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
    math(EXPR VERSION_NUMBER "${VERSION_MAJOR} * 10000 + ${VERSION_MINOR} * 100 + ${VERSION_PATCH}")
    set(${version_number} "${VERSION_NUMBER}" PARENT_SCOPE)
endfunction()

if(SYCL_COMPILER)
    parse_icx_version(SYCL_COMPILER_VERSION)
endif()

if(NOT SYCL_COMPILER_VERSION)
    set(SYCL_FOUND FALSE)
    set(SYCL_REASON_FAILURE "Cannot parse SYCL compiler version")
    find_package_handle_standard_args(SYCL DEFAULT_MSG SYCL_FOUND)
    return()
endif()

# -- Find include directory --
find_file(SYCL_INCLUDE_DIR_TMP
    NAMES include
    HINTS ${SYCL_ROOT}
    NO_DEFAULT_PATH
)

find_file(SYCL_INCLUDE_SYCL_DIR
    NAMES sycl
    HINTS ${SYCL_ROOT}/include/
    NO_DEFAULT_PATH
)

set(SYCL_INCLUDE_DIR ${SYCL_INCLUDE_DIR_TMP})
if(SYCL_INCLUDE_SYCL_DIR)
    list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})
endif()

# -- Find library directory --
find_file(SYCL_LIBRARY_DIR
    NAMES lib
    HINTS ${SYCL_ROOT}
    NO_DEFAULT_PATH
)

# -- Find SYCL library --
find_library(SYCL_LIBRARY
    NAMES sycl
    HINTS ${SYCL_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

if(NOT SYCL_LIBRARY)
    set(SYCL_FOUND FALSE)
    set(SYCL_REASON_FAILURE "SYCL library (sycl) not found")
    find_package_handle_standard_args(SYCL DEFAULT_MSG SYCL_FOUND)
    return()
endif()

set(SYCL_FOUND TRUE)
find_package_handle_standard_args(SYCL
    FOUND_VAR SYCL_FOUND
    REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY
    REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}"
    VERSION_VAR SYCL_COMPILER_VERSION
)

mark_as_advanced(SYCL_COMPILER SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY SYCL_COMPILER_VERSION)
