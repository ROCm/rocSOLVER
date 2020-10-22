# ########################################################################
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
# ########################################################################

# For debugging, uncomment this
# set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O0" )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.65.1 1.65 )
set( Boost_USE_STATIC_LIBS OFF )

if(EXISTS /etc/redhat-release)
    if(CXX_VERSION_STRING MATCHES "clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread -stdlib=libstdc++ -std=c++14")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
endif()

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
    # triangular factorizations and linear solvers
    getrs_gtest.cpp
    getri_gtest.cpp
    getf2_getrf_gtest.cpp
    potf2_potrf_gtest.cpp
    # bidiagonal matrices and svd
    gebd2_gebrd_gtest.cpp
    gesvd_gtest.cpp
    labrd_gtest.cpp
    bdsqr_gtest.cpp
    # orthogonal factorizations
    geqr2_geqrf_gtest.cpp
    geql2_geqlf_gtest.cpp
    gelq2_gelqf_gtest.cpp
    # orthonormal/unitary matrices
    orgxr_ungxr_gtest.cpp
    orglx_unglx_gtest.cpp
    orgxl_ungxl_gtest.cpp
    orgbr_ungbr_gtest.cpp
    orgtr_ungtr_gtest.cpp
    ormxr_unmxr_gtest.cpp
    ormlx_unmlx_gtest.cpp
    ormxl_unmxl_gtest.cpp
    ormbr_unmbr_gtest.cpp
    ormtr_unmtr_gtest.cpp
    # householder reflections
    larf_gtest.cpp
    larfg_gtest.cpp
    larft_gtest.cpp
    larfb_gtest.cpp
    # vector & matrix manipulations
    lacgv_gtest.cpp
    laswp_gtest.cpp
    )

set(rocsolver_test_source
    rocsolver_gtest_main.cpp
)

set( rocsolver_benchmark_common
      ../../../rocblascommon/clients/common/utility.cpp
      ../../../rocblascommon/library/src/rocblas_ostream.cpp
      ../common/cblas_interface.cpp
    )

add_executable( rocsolver-test ${roclapack_test_source} ${rocsolver_test_source} ${rocsolver_benchmark_common} )

add_armor_flags( rocsolver-test "${ARMOR_LEVEL}" )

target_compile_definitions( rocsolver-test PRIVATE BUILD_WITH_TENSILE=0 GOOGLE_TEST )

# Internal header includes
target_include_directories( rocsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)

#set( BLIS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/build/deps/blis/include/blis )
#set( BLIS_LIBRARY ${CMAKE_SOURCE_DIR}/build/deps/blis/lib/libblis.so )

if( OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    if( OS_ID_rhel OR OS_ID_centos)
        if( EXISTS "/usr/lib/gcc/x86_64-redhat-linux/8/" )
            set( ROCM_OPENMP_PATH /usr/lib/gcc/x86_64-redhat-linux/8 )
        else()
            set( ROCM_OPENMP_PATH /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7 )
        endif()

        # defer OpenMP include as search order must come after clang
        set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        set( OPENMP_LIBRARY ${ROCM_OPENMP_PATH}/libgomp.so )
        if( CXX_VERSION_STRING MATCHES "clang")
            set( XXX_OPENMP_INCLUDE_DIR ${ROCM_OPENMP_PATH}/include )
        endif()
    else()
    #SLES
        set( XXX_OPENMP_INCLUDE_DIR /usr/lib64/gcc/x86_64-suse-linux/7/include/ )
        set( OPENMP_LIBRARY /usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so )
    endif()

    message(STATUS "RocmPath: ${ROCM_PATH}")
    if(EXISTS ${ROCM_PATH}/llvm/lib/clang/12.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/llvm/lib/clang/12.0.0/include )
    elseif(EXISTS ${ROCM_PATH}/llvm/lib/clang/11.0.0/include/immintrin.h AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/llvm/lib/clang/11.0.0/include )
    elseif( EXISTS "${ROCM_PATH}/hcc/lib/clang/10.0.0/include/immintrin.h" AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/10.0.0/include )
    elseif( EXISTS "${ROCM_PATH}/hcc/lib/clang/9.0.0/include/immintrin.h" AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
        set( CLANG_INCLUDE_DIR ${ROCM_PATH}/hcc/lib/clang/9.0.0/include )
    else()
        set( CLANG_INCLUDE_DIR )
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

target_link_libraries( rocsolver-test PRIVATE roc::rocblas ) #${ROCBLAS_LIBRARY})

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
  target_link_libraries( rocsolver-test PRIVATE hip::device )
endif( )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem ${ROCM_PATH}/include'"
  target_compile_options( rocsolver-test PRIVATE -Wno-unused-command-line-argument -mf16c )
  target_include_directories( rocsolver-test PRIVATE ${ROCM_PATH}/hsa/include)
elseif( CMAKE_COMPILER_IS_GNUCXX OR CXX_VERSION_STRING MATCHES "clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( rocsolver-test PRIVATE -mf16c )
endif( )

if( CXX_VERSION_STRING MATCHES "clang" )
  target_link_libraries( rocsolver-test PRIVATE -lpthread -lstdc++ -lm )
  if(OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
else( )
  if(OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    set(CMAKE_CXX_FLAGS "-isystem ${CLANG_INCLUDE_DIR} -isystem ${XXX_OPENMP_INCLUDE_DIR} ${CMAKE_CXX_FLAGS}")
  endif( )
  set(CMAKE_CXX_FLAGS "-isystem ${ROCM_PATH}/include ${CMAKE_CXX_FLAGS}")
endif( )

set_target_properties( rocsolver-test PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocsolver-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/../../clients/staging" )
target_compile_definitions( rocsolver-test PRIVATE ROCM_USE_FLOAT16 )
