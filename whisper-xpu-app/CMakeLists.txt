add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/sycl_src
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)

target_compile_options(whisper_xpu_app PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8 /EHa>)

target_link_libraries(whisper_xpu_app PRIVATE
    whisper_xpu_core
    wx::core
    portaudio
)

# Delay-load sycl8.dll so we can set ZES_ENABLE_SYSMAN=1 before the
# SYCL runtime initializes (required by Level Zero loader on some
# Intel GPU driver versions to avoid a null-function-pointer crash).
if(WIN32)
    target_link_options(whisper_xpu_app PRIVATE /DELAYLOAD:sycl8.dll)
    target_link_libraries(whisper_xpu_app PRIVATE delayimp)
endif()

if(WIN32)
    target_link_libraries(whisper_xpu_app PRIVATE
        winmm
        ole32
        uuid
        comctl32
        rpcrt4
    )
endif()
