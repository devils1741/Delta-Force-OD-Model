cmake_minimum_required(VERSION 4.2)
project(delta LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "C:/Program Files/TensorRT-10.16.1.11" CACHE PATH "TensorRT installation directory")

find_package(CUDAToolkit REQUIRED)

add_executable(delta
        main.cpp
        Detection.cpp
        Detection.h
        CudaPreprocess.cu
        CudaPreprocess.h
        DxgiScreenCapture.cpp
        DxgiScreenCapture.h
        OverlayWindow.cpp
        OverlayWindow.h
        RealtimePipeline.cpp
        RealtimePipeline.h
        ScreenCapture.cpp
        ScreenCapture.h
        TensorRtDetector.cpp
        TensorRtDetector.h)

target_compile_options(delta PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

target_include_directories(delta PRIVATE
        "${TENSORRT_ROOT}/include")

target_link_directories(delta PRIVATE
        "${TENSORRT_ROOT}/lib")

target_link_libraries(delta PRIVATE
        CUDA::cudart_static
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
