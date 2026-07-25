# Google Test
set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/third_party/gtest")
if(NOT TARGET GTest::gtest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${GTEST_ROOT} build-gtest EXCLUDE_FROM_ALL)
endif()

add_subdirectory(sycl_device_detect)

# VAD benchmark (links against engine + whisper)
add_executable(bench_vad bench_vad.cpp)
target_include_directories(bench_vad PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
target_link_libraries(bench_vad PRIVATE whisper_xpu_core)

enable_testing()
