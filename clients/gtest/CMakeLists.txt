# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# For debugging, uncomment this
# set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

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

find_package( GTest REQUIRED )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

set(roclapack_test_source
    getf2_getrf_gtest.cpp
    getf2_getrf_batched_gtest.cpp
    getf2_getrf_strided_batched_gtest.cpp
    larfg_gtest.cpp
    larf_gtest.cpp
    geqr2_geqrf_gtest.cpp
    geqr2_geqrf_batched_gtest.cpp
    geqr2_geqrf_strided_batched_gtest.cpp
    getrs_gtest.cpp
    potf2_gtest.cpp
    )

set(rocsolver_test_source
    rocsolver_gtest_main.cpp
)

set( rocsolver_benchmark_common
      ../common/utility.cpp
      ../common/cblas_interface.cpp
      ../common/norm.cpp
      ../common/unit.cpp
      ../common/near.cpp
      ../common/arg_check.cpp
    )

add_executable( rocsolver-test ${roclapack_test_source} ${rocsolver_test_source} ${rocsolver_benchmark_common} )
#target_compile_features( rocsolver-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )

target_compile_definitions( rocsolver-test PRIVATE BUILD_WITH_TENSILE=0 GOOGLE_TEST )

# Internal header includes
target_include_directories( rocsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)


#set( BLIS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/build/deps/blis/include/blis )
#set( BLIS_LIBRARY ${CMAKE_SOURCE_DIR}/build/deps/blis/lib/libblis.so )

if( OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    if( OS_ID_rhel OR OS_ID_centos)
        # defer OpenMP include as search order must come after clang
        set( XXX_OPENMP_INCLUDE_DIR /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include )
        set( OPENMP_LIBRARY /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/libgomp.so )
    else()
    #SLES
        set( OPENMP_INCLUDE_DIR /usr/lib64/gcc/x86_64-suse-linux/7/include/ )
        set( OPENMP_LIBRARY /usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so )
    endif()

    if(EXISTS /opt/rocm/hcc/lib/clang/10.0.0/include/immintrin.h)
        set( CLANG_INCLUDE_DIR /opt/rocm/hcc/lib/clang/10.0.0/include )
    elseif (EXISTS /opt/rocm/hcc/lib/clang/9.0.0/include/immintrin.h)
        set( CLANG_INCLUDE_DIR /opt/rocm/hcc/lib/clang/9.0.0/include )
    else()
        error("cannot find immintrin.h")
    endif()

    # External header includes included as system files
    target_include_directories( rocsolver-test
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${CLANG_INCLUDE_DIR}> 
#        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${OPENMP_INCLUDE_DIR}> 
        )

    target_link_libraries( rocsolver-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENMP_LIBRARY} cblas lapack roc::rocsolver )

else()
    # External header includes included as system files
    target_include_directories( rocsolver-test
      SYSTEM PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${HCC_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
#        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
        )

    target_link_libraries( rocsolver-test PRIVATE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} cblas lapack roc::rocsolver )
endif()

target_link_libraries( rocsolver-test PRIVATE /opt/rocm/rocblas/lib/librocblas.so ) #${ROCBLAS_LIBRARY})


get_target_property( HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )

if( CUDA_FOUND )
  target_include_directories( rocsolver-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocsolver-test PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocsolver-test PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_compile_definitions( rocsolver-test PRIVATE __HIP_PLATFORM_HCC__ )
  target_link_libraries( rocsolver-test PRIVATE ${HIPHCC_LOCATION} )
endif( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocsolver-test PRIVATE -Wno-unused-command-line-argument -mf16c )
  target_include_directories( rocsolver-test PRIVATE /opt/rocm/hsa/include)
elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang" )
  # GCC or hip-clang needs specific flag to turn on f16c intrinsics
  target_compile_options( rocsolver-test PRIVATE -mf16c )
endif( )

if( OS_ID_rhel OR OS_ID_centos)
    # force clang includes to take precedence over devtoolset-7 which we only want for OpenMP
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
endif()

if( CXX_VERSION_STRING MATCHES "clang" )
  target_link_libraries( rocsolver-test PRIVATE -lpthread -lm )
endif( )

set_target_properties( rocsolver-test PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocsolver-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
