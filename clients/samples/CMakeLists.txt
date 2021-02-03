# ########################################################################
# Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
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
add_executable( example-c-hmm
  example_hmm.c
)
add_executable( example-cpp-logging
  example_logging.cpp
)

# group sample programs by language
set( c_samples
  example-c-basic
  example-c-hmm
)
set( cpp_samples
  example-cpp-basic
  example-cpp-logging
)
set( fortran_samples
  example-fortran-basic
)

# set flags for building the sample programs
foreach( exe ${c_samples} ${cpp_samples} ${fortran_samples} )
  target_link_libraries( ${exe} PRIVATE roc::rocsolver roc::rocblas )

  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )

  if( CUDA_FOUND )
    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )
    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  endif( )

  if( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
  endif( )

  if( NOT CUDA_FOUND )
    target_link_libraries( ${exe} PRIVATE hip::host )
  endif()
endforeach( )

foreach( exe ${cpp_samples} )
  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endforeach( )

foreach( exe ${c_samples} )
  set_target_properties( ${exe} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endforeach( )
