cmake_minimum_required(VERSION 4.2)
# RTX 4070 SUPER: emit native SM89 code to avoid PTX JIT driver requirements.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89-real)
endif()
project(delta LANGUAGES CXX CUDA)

# Embed the MSVC and CUDA runtimes in all targets, including local dependencies.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp statically" FORCE)
set(YAML_MSVC_SHARED_RT OFF CACHE BOOL "Use the static MSVC runtime for yaml-cpp" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "C:/Program Files/TensorRT-11.4.0.106" CACHE PATH "TensorRT installation directory")

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
# Prefer the bundled source so its runtime linkage matches the application.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp-0.8.0/CMakeLists.txt")
    find_package(yaml-cpp CONFIG QUIET)
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp-0.8.0/CMakeLists.txt" OR NOT yaml-cpp_FOUND)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp-0.8.0/CMakeLists.txt")
        FetchContent_Declare(yaml-cpp
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp-0.8.0")
    else()
    FetchContent_Declare(
            yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG 0.8.0)
    endif()
    FetchContent_MakeAvailable(yaml-cpp)
endif()

add_executable(delta
        src/main.cpp
        src/source/MouseController.cpp
        src/source/AppConfig.cpp
        src/source/CudaPostprocess.cu
        src/source/CudaPreprocess.cu
        src/source/DxgiScreenCapture.cpp
        src/source/OverlayWindow.cpp
        src/source/RealtimePipeline.cpp
        src/source/TensorRtDetector.cpp
        src/include/MouseController.h
        src/include/AppConfig.h
        src/include/Detection.h
        src/include/CudaPostprocess.h
        src/include/CudaPreprocess.h
        src/include/DxgiScreenCapture.h
        src/include/OverlayWindow.h
        src/include/RealtimePipeline.h
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
        nvinfer_11
        nvonnxparser_11
        nvinfer_plugin_11
        user32
        gdi32
        d3d11
        dxgi
        dcomp
        d3dcompiler)

# Keep runtime configuration beside the executable for Explorer/Release launches.
add_custom_command(TARGET delta POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:delta>/config"
        COMMAND ${CMAKE_COMMAND}
        "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/config/config.yaml"
        "-DDESTINATION=$<TARGET_FILE_DIR:delta>/config/config.yaml"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyDefaultConfig.cmake"
        VERBATIM)

foreach(tensorrt_dll IN ITEMS
        nvinfer_11.dll
        nvinfer_plugin_11.dll
        nvonnxparser_11.dll
        nvinfer_builder_resource_sm89_11.dll
        nvinfer_builder_resource_ptx_11.dll)
    add_custom_command(TARGET delta POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_ROOT}/bin/${tensorrt_dll}"
            "$<TARGET_FILE_DIR:delta>/${tensorrt_dll}")
endforeach()

find_package(Doxygen QUIET)

if (DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
            "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
            @ONLY)

    add_custom_target(docs
            COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Generate API documentation with Doxygen"
            VERBATIM)
else()
    message(STATUS "Doxygen not found; install Doxygen to enable the docs target.")
endif()

include(CTest)
add_subdirectory(ut/test_mouse_controll)

if(BUILD_TESTING)
    add_subdirectory(ut/test_gpu_pipeline)
endif()
