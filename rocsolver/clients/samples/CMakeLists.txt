# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( example-simple
  example_simple.cpp
)

set( c_samples
)
set( cpp_samples
  example-simple
)
set( fortran_samples
)

foreach( exe ${c_samples} ${cpp_samples} ${fortran_samples} )
  target_link_libraries( ${exe} PRIVATE roc::rocsolver roc::rocblas )

  set_target_properties( ${exe} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
  )

  if( CUDA_FOUND )
    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  else( )
    # auto set in hip_common.h
    #target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_HCC__ )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
    # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
    # "clang-5.0: warning: argument unused during compilation: '-isystem ${ROCM_PATH}/include'"
    # include order workaround to force /opt/rocm/include later in order to ignore installed rocblas
    set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS}")
    target_compile_options( ${exe} PRIVATE -Wno-unused-command-line-argument )
    target_include_directories( ${exe} PRIVATE ${ROCM_PATH}/hsa/include)
  elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
  endif( )

  if( NOT CUDA_FOUND )
    target_link_libraries( ${exe} PRIVATE hip::host )
  endif()
endforeach( )

foreach( exe ${c_samples} ${cpp_samples} )
  if( NOT CUDA_FOUND )
    target_link_libraries( ${exe} PRIVATE hip::device )
  endif()
endforeach( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # include order workaround to force /opt/rocm/include later in order to ignore installed rocblas
  set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS}")
endif( )
