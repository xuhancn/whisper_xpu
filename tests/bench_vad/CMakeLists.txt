add_executable(bench_vad bench_vad.cpp)
target_include_directories(bench_vad PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
target_link_libraries(bench_vad PRIVATE whisper_xpu_core)
