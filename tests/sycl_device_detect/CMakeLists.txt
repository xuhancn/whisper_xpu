# ── Non-SYCL basics test (MSVC, no SYCL runtime needed) ──
add_executable(test_sycl_device_detect
    test_sycl_device_detect.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src/device_detect.cpp
)
target_include_directories(test_sycl_device_detect PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
# Remove global WHISPER_XPU_HAS_SYCL for the no-SYCL build
get_target_property(_defs test_sycl_device_detect COMPILE_DEFINITIONS)
if(_defs)
    list(REMOVE_ITEM _defs "WHISPER_XPU_HAS_SYCL=1")
    set_target_properties(test_sycl_device_detect PROPERTIES COMPILE_DEFINITIONS "${_defs}")
endif()
target_link_libraries(test_sycl_device_detect PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME sycl_device_detect/basics COMMAND test_sycl_device_detect)

# ── SYCL test (compiled with icpx, linked against sycl DLL import lib) ──
if(HAVE_SYCL)
    set(TEST_SYCL_SOURCES test_sycl_device_detect.cpp)
    set(TEST_SYCL_INC_DIRS
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core
        ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
        ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
        ${SYCL_INCLUDE_DIR}
    )

    add_executable(test_sycl_device_detect_sycl ${TEST_SYCL_SOURCES})
    target_include_directories(test_sycl_device_detect_sycl PRIVATE ${TEST_SYCL_INC_DIRS})
    target_compile_definitions(test_sycl_device_detect_sycl PRIVATE WHISPER_XPU_HAS_SYCL)
    target_compile_options(test_sycl_device_detect_sycl PRIVATE -fsycl)

    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
        if(DEFINED ENV{DNNL_INTEL_TOOLSET})
            set_target_properties(test_sycl_device_detect_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "$ENV{DNNL_INTEL_TOOLSET}")
        else()
            set_target_properties(test_sycl_device_detect_sycl PROPERTIES
                VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        endif()
    endif()

    target_link_libraries(test_sycl_device_detect_sycl PRIVATE
        GTest::gtest GTest::gtest_main
        whisper_xpu_sycl_core
        sycl
    )
    if(SYCL_LIBRARY_DIR)
        target_link_directories(test_sycl_device_detect_sycl PRIVATE ${SYCL_LIBRARY_DIR})
    endif()
    add_test(NAME sycl_device_detect/sycl COMMAND test_sycl_device_detect_sycl)
endif()
