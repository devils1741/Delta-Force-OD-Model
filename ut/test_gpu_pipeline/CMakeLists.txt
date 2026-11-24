add_executable(gpu_pipeline_test
    main.cpp
    "${PROJECT_SOURCE_DIR}/src/source/RealtimePipeline.cpp"
    "${PROJECT_SOURCE_DIR}/src/source/MouseController.cpp"
    "${PROJECT_SOURCE_DIR}/src/source/CudaPostprocess.cu"
    "${PROJECT_SOURCE_DIR}/src/source/OverlayWindow.cpp")
target_compile_definitions(gpu_pipeline_test PRIVATE DELTA_GPU_TESTING)
target_include_directories(gpu_pipeline_test PRIVATE "${PROJECT_SOURCE_DIR}/src/include")
target_link_libraries(gpu_pipeline_test PRIVATE CUDA::cudart_static d3d11 dxgi dcomp d3dcompiler user32)
target_compile_options(gpu_pipeline_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
add_test(NAME gpu_pipeline COMMAND gpu_pipeline_test)
set_tests_properties(gpu_pipeline PROPERTIES TIMEOUT 30)
