# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.65.1 1.65 )
set( Boost_USE_STATIC_LIBS OFF )

find_package( Boost COMPONENTS program_options )

if( NOT Boost_FOUND )
  message( STATUS "Dynamic boost libraries not found. Attempting to find static libraries " )
  set( Boost_USE_STATIC_LIBS ON )
  find_package( Boost COMPONENTS program_options )

  if( NOT Boost_FOUND )
    message( FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH" )
  endif( )
endif( )

# Linking lapack library requires fortran flags
enable_language( Fortran )
find_package( cblas CONFIG REQUIRED )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

set( rocsolver_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/near.cpp
      ../common/arg_check.cpp
    )

add_executable( rocsolver-bench client.cpp ${rocsolver_benchmark_common} )
target_compile_features( rocsolver-bench PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

rocm_clang_tidy_check(rocsolver-bench)

if( BUILD_WITH_TENSILE )
    target_compile_definitions( rocsolver-bench PRIVATE BUILD_WITH_TENSILE=1 )
else()
    target_compile_definitions( rocsolver-bench PRIVATE BUILD_WITH_TENSILE=0 )
endif()

# Internal header includes
target_include_directories( rocsolver-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories( rocsolver-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    )

target_link_libraries( rocsolver-bench PRIVATE ${Boost_LIBRARIES} cblas lapack roc::rocsolver )
target_link_libraries( rocsolver-bench PRIVATE /opt/rocm/rocblas/lib/librocblas.so ) #${ROCBLAS_LIBRARY})

get_target_property( HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )

if( CUDA_FOUND )
  target_include_directories( rocsolver-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocsolver-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocsolver-bench PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_compile_definitions( rocsolver-bench PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocsolver-bench PRIVATE ${HIPHCC_LOCATION} )
endif( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocsolver-bench PRIVATE -Wno-unused-command-line-argument -mf16c )

elseif( CMAKE_COMPILER_IS_GNUCXX )
  # GCC needs specific flags to turn on f16c intrinsics
  target_compile_options( rocsolver-bench PRIVATE -mf16c )
endif( )

set_target_properties( rocsolver-bench PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocsolver-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
