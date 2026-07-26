add_executable(bench_vad bench_vad.cpp)
target_include_directories(bench_vad PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)
target_compile_definitions(bench_vad PRIVATE
    TEST_WAV_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    VAD_MODEL_DIR="${CMAKE_SOURCE_DIR}/models/vad_model"
)
target_link_libraries(bench_vad PRIVATE whisper_xpu_cpu_core whisper_xpu_sycl_core)
