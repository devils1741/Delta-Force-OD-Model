cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(test_mouse_controll LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
    message(FATAL_ERROR "This mouse test requires Windows")
endif()

get_filename_component(MODEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

add_executable(mouse_rectangle_test
    main.cpp
    MouseRectangleTest.h
    MouseRectangleTest.cpp
    "${MODEL_ROOT}/src/source/MouseController.cpp"
    "${MODEL_ROOT}/src/include/MouseController.h")
target_include_directories(mouse_rectangle_test PRIVATE "${MODEL_ROOT}/src/include")
if(MSVC)
    target_compile_options(mouse_rectangle_test PRIVATE /utf-8)
endif()
target_link_libraries(mouse_rectangle_test PRIVATE user32 gdi32)

add_executable(mouse_input_probe
    probe_main.cpp
    MouseInputProbe.h
    MouseInputProbe.cpp
    "${MODEL_ROOT}/src/source/MouseController.cpp"
    "${MODEL_ROOT}/src/include/MouseController.h")
target_include_directories(mouse_input_probe PRIVATE "${MODEL_ROOT}/src/include")
if(MSVC)
    target_compile_options(mouse_input_probe PRIVATE /utf-8)
endif()
target_link_libraries(mouse_input_probe PRIVATE user32)
