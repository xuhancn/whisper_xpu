add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/whisper.cpp/include
)

target_include_directories(whisper_xpu_app PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/portaudio/include
)

target_link_libraries(whisper_xpu_app PRIVATE
    whisper_xpu_core
    wx::core          # wxWidgets GUI library (depends on wx::base)
    portaudio
)

if(WIN32)
    target_link_libraries(whisper_xpu_app PRIVATE
        winmm
        ole32
        uuid
        comctl32
        rpcrt4
    )
endif()
