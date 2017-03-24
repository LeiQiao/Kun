cmake_minimum_required (VERSION 2.8)

cmake_policy(SET CMP0015 NEW)

project(%MODEL_NAME%)

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

set(CMAKE_CXX_FLAGS "-DDEBUG -DSAVE_UNIT_TEST_TENSOR ${CMAKE_CXX_FLAGS}")

include_directories(./eigen ./src ./supports)
add_executable(%MODEL_NAME% ./main.cpp ./supports/conv3x3_neon.cpp ./supports/EigenCustomOp.cpp ./src/%MODEL_NAME%.cpp)