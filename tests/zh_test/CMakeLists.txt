# Standalone Chinese-display converter test.  Compiles zh_converter.cpp +
# zh_t2s.cpp directly (no DLL) and checks Traditional↔Simplified on known
# text.  No SYCL/wx/portaudio dependency — pure C++ over the vendored OpenCC
# text dicts.
add_executable(zh_test
    zh_test.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app/zh_converter.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/zh_t2s.cpp
)
target_include_directories(zh_test PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core)
# The converter reads the vendored OpenCC dicts from this absolute path.
target_compile_definitions(zh_test PRIVATE
    WHISPER_XPU_OPENCC_DATA_DIR="${CMAKE_SOURCE_DIR}/third_party/opencc-data")
if(MSVC)
    target_compile_options(zh_test PRIVATE /utf-8)
endif()
