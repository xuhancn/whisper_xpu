add_executable(test_ring_buffer test_ring_buffer.cpp)
target_include_directories(test_ring_buffer PRIVATE
    ${CMAKE_SOURCE_DIR}/whisper-xpu-app
)
target_link_libraries(test_ring_buffer PRIVATE GTest::gtest GTest::gtest_main)
add_test(NAME ring_buffer COMMAND test_ring_buffer)
