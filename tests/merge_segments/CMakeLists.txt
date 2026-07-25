add_executable(test_merge_segments
    test_merge_segments.cpp
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core/merge_segments.cpp
)
target_include_directories(test_merge_segments PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-core
)
target_link_libraries(test_merge_segments PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME merge_segments COMMAND test_merge_segments)
