add_executable(whisper_xpu_app WIN32
    main.cpp
    app_frame.cpp
    audio_capture.cpp
    src/audio_stream.cpp
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
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

target_link_libraries(whisper_xpu_app PRIVATE
    whisper_xpu_sycl_core
    wx::core
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

# ── Copy project DLL next to the executable ──
add_custom_command(TARGET whisper_xpu_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:whisper_xpu_sycl_core>
        $<TARGET_FILE_DIR:whisper_xpu_app>
    COMMENT "Copying whisper_xpu_sycl_core.dll to app output directory"
)

# ── Copy Intel oneAPI runtime DLLs ──
# These are needed by whisper_xpu_sycl_core.dll at runtime.
# ze_loader.dll is NOT copied — it's part of the Intel GPU driver
# and must be present on the target system for GPU functionality.
if(WIN32 AND SYCL_LIBRARY_DIR)
    get_filename_component(_sycl_bin "${SYCL_LIBRARY_DIR}/../bin" ABSOLUTE)
    set(_rt_dlls sycl8.dll svml_dispmd.dll libmmd.dll libiomp5md.dll
        ur_win_proxy_loader.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_adapter_opencl.dll
        umf.dll tcm.dll)
    # Optional DLLs — may not exist in older oneAPI releases
    foreach(_opt_dll ur_adapter_level_zero_v2.dll libhwloc-15.dll)
        if(EXISTS "${_sycl_bin}/${_opt_dll}")
            list(APPEND _rt_dlls ${_opt_dll})
        endif()
    endforeach()
    foreach(_dll ${_rt_dlls})
        add_custom_command(TARGET whisper_xpu_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_sycl_bin}/${_dll}"
                $<TARGET_FILE_DIR:whisper_xpu_app>
            COMMENT "Copying ${_dll} to app output directory"
        )
    endforeach()
endif()
