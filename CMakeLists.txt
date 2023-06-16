cmake_minimum_required(VERSION 3.10)
project(catflow-edge)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cli11/)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Library
add_library(libcatflow ${PROJECT_SOURCE_DIR}/src/yolo.cpp)

# Main executable
add_executable(catflow-edge "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(catflow-edge libcatflow ${OpenCV_LIBS})

# Tests
enable_testing()

set(TEST_VECTOR "${PROJECT_SOURCE_DIR}/test/wikipedia_cc_640x480.png")
set(TEST_MODEL "${PROJECT_SOURCE_DIR}/test/yolov5m.onnx")

add_executable(catflow-edge-test "${PROJECT_SOURCE_DIR}/test/test_inference.cpp")
target_link_libraries(catflow-edge-test libcatflow ${OpenCV_LIBS})

add_test(NAME test_inference COMMAND catflow-edge-test ${TEST_MODEL} ${TEST_VECTOR})
