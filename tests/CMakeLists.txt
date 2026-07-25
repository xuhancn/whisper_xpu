# Google Test
set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/third_party/gtest")
if(NOT TARGET GTest::gtest)
    add_subdirectory(${GTEST_ROOT} build-gtest EXCLUDE_FROM_ALL)
endif()

add_subdirectory(sycl_device_detect)
enable_testing()
