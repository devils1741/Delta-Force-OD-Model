cmake_minimum_required(VERSION 4.2)
project(delta LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "C:/Program Files/TensorRT-10.16.1.11" CACHE PATH "TensorRT installation directory")

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp CONFIG QUIET)

if (NOT yaml-cpp_FOUND)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG 0.8.0)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

add_executable(delta
        src/main.cpp
        src/source/AppConfig.cpp
        src/source/Detection.cpp
        src/source/CudaPreprocess.cu
        src/source/DxgiScreenCapture.cpp
        src/source/OverlayWindow.cpp
        src/source/PreviewWindow.cpp
        src/source/RealtimePipeline.cpp
        src/source/ScreenCapture.cpp
        src/source/TensorRtDetector.cpp
        src/include/AppConfig.h
        src/include/Detection.h
        src/include/CudaPreprocess.h
        src/include/DxgiScreenCapture.h
        src/include/OverlayWindow.h
        src/include/PreviewWindow.h
        src/include/RealtimePipeline.h
        src/include/ScreenCapture.h
        src/include/TensorRtDetector.h)

target_compile_options(delta PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

target_include_directories(delta PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
        "${TENSORRT_ROOT}/include")

target_link_directories(delta PRIVATE
        "${TENSORRT_ROOT}/lib")

target_link_libraries(delta PRIVATE
        CUDA::cudart_static
        yaml-cpp::yaml-cpp
        nvinfer_10
        nvonnxparser_10
        nvinfer_plugin_10
        user32
        gdi32
        d3d11
        dxgi)

foreach(tensorrt_dll IN ITEMS
        nvinfer_10.dll
        nvinfer_plugin_10.dll
        nvonnxparser_10.dll
        nvinfer_builder_resource_sm89_10.dll
        nvinfer_builder_resource_ptx_10.dll)
    add_custom_command(TARGET delta POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_ROOT}/bin/${tensorrt_dll}"
            "$<TARGET_FILE_DIR:delta>/${tensorrt_dll}")
endforeach()
