add_library(whisper_xpu_core STATIC
    engine.cpp
)

target_include_directories(whisper_xpu_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# whisper.cpp headers
target_include_directories(whisper_xpu_core PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/ggml/include
)

target_link_libraries(whisper_xpu_core PUBLIC
    whisper    # from whisper.cpp's CMake
)

if(HAVE_SYCL)
    target_compile_definitions(whisper_xpu_core PRIVATE WHISPER_XPU_HAS_SYCL)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_core PRIVATE winmm)
endif()
