cmake_minimum_required (VERSION 2.8)

cmake_policy(SET CMP0015 NEW)

project(mnist_predict)

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

set(CMAKE_CXX_FLAGS "-DDEBUG -DSAVE_UNIT_TEST_TENSOR ${CMAKE_CXX_FLAGS}")

IF(APPLE)
    FIND_LIBRARY(ACCELERATE_LIBRARY Accelerate)
    MARK_AS_ADVANCED(ACCELERATE_LIBRARY)
ENDIF (APPLE)

link_libraries(opencv_core opencv_imgproc opencv_imgcodecs ${ACCELERATE_LIBRARY})

include_directories(./eigen ./cpp ../UnitTest/EigenCustomCode)
add_executable(mnist_predict ./main.cpp ../UnitTest/EigenCustomCode/conv3x3_neon.cpp ../UnitTest/EigenCustomCode/EigenCustomOp.cpp ./cpp/mnist.cpp)