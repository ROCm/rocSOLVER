# ########################################################################
# Copyright (c) 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# the interface to rocblas for Fortran programs
add_library( rocblas_module OBJECT
  "${ROCBLAS_INCLUDE_DIR}/rocblas_module.f90"
)

# declare sample programs
add_executable( example-c-basic
  example_basic.c
)
add_executable( example-cpp-basic
  example_basic.cpp
)
add_executable( example-fortran-basic
  example_basic.f90
  $<TARGET_OBJECTS:rocblas_module>
)

# group sample programs by language
set( c_samples
  example-c-basic
)
set( cpp_samples
  example-cpp-basic
)
set( fortran_samples
  example-fortran-basic
)

# set flags for building the sample programs
foreach( exe ${c_samples} ${cpp_samples} ${fortran_samples} )
  target_link_libraries( ${exe} PRIVATE roc::rocsolver roc::rocblas )

  set_target_properties( ${exe} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

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

foreach( exe ${cpp_samples} )
  if( NOT CUDA_FOUND )
    target_link_libraries( ${exe} PRIVATE hip::device )
  endif()
endforeach( )

foreach( exe ${c_samples} )
  set_target_properties( ${exe} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endforeach( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # include order workaround to force /opt/rocm/include later in order to ignore installed rocblas
  set(CMAKE_CXX_FLAGS "-isystem /opt/rocm/include ${CMAKE_CXX_FLAGS}")
endif( )
