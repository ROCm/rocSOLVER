# ########################################################################
# Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
# ########################################################################

find_package(GTest REQUIRED)

set(roclapack_test_source
  # linear systems solvers
  getri_gtest.cpp
  getrs_gtest.cpp
  gesv_gtest.cpp
  potrs_gtest.cpp
  posv_gtest.cpp
  potri_gtest.cpp
  trtri_gtest.cpp
  # least squares solvers
  gels_gtest.cpp
  # triangular factorizations
  getf2_getrf_gtest.cpp
  potf2_potrf_gtest.cpp
  sytf2_sytrf_gtest.cpp
  # orthogonal factorizations
  geqr2_geqrf_gtest.cpp
  gerq2_gerqf_gtest.cpp
  geql2_geqlf_gtest.cpp
  gelq2_gelqf_gtest.cpp
  # problem and matrix reductions (diagonalizations)
  gebd2_gebrd_gtest.cpp
  sytxx_hetxx_gtest.cpp
  sygsx_hegsx_gtest.cpp
  # singular value decomposition
  gesvd_gtest.cpp
  gesvdx_gtest.cpp
  # symmetric eigensolvers
  syev_heev_gtest.cpp
  syevd_heevd_gtest.cpp
  syevj_heevj_gtest.cpp
  syevx_heevx_gtest.cpp
  syevdx_heevdx_gtest.cpp
  sygv_hegv_gtest.cpp
  sygvd_hegvd_gtest.cpp
  sygvj_hegvj_gtest.cpp
  sygvx_hegvx_gtest.cpp
  sygvdx_hegvdx_gtest.cpp
)

set(rocauxiliary_test_source
  # vector & matrix manipulations
  lacgv_gtest.cpp
  laswp_gtest.cpp
  # householder reflections
  larf_gtest.cpp
  larfg_gtest.cpp
  larft_gtest.cpp
  larfb_gtest.cpp
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
  # bidiagonal matrices
  labrd_gtest.cpp
  bdsqr_gtest.cpp
  bdsvdx_gtest.cpp
  # tridiagonal matrices
  sterf_gtest.cpp
  steqr_gtest.cpp
  stedc_gtest.cpp
  stebz_gtest.cpp
  stein_gtest.cpp
  latrd_gtest.cpp
  # symmetric matrices
  lasyf_gtest.cpp
)

set(others_test_source
  # unified memory model
  managed_malloc_gtest.cpp
  # rocblas memory model
  memory_model_gtest.cpp
  # rocsolver logging
  logging_gtest.cpp
  # helpers
  client_environment_helpers.cpp
)

set(rocsolver_test_source
  rocsolver_gtest_main.cpp
)

add_executable(rocsolver-test
  ${roclapack_test_source}
  ${rocauxiliary_test_source}
  ${others_test_source}
  ${rocsolver_test_source}
)

add_armor_flags(rocsolver-test "${ARMOR_LEVEL}")

if(WIN32)
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${ROCSOLVER_LAPACK_PATH}/bin/*.dll
    ${GTest_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET rocsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/
    )
  endforeach()
  add_custom_command(TARGET rocsolver-test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory $ENV{rocblas_DIR}/bin/rocblas/library ${PROJECT_BINARY_DIR}/staging/library
  )
endif()

target_link_libraries(rocsolver-test PRIVATE
  GTest::GTest
  hip::device
  rocsolver-common
  clients-common
  $<$<PLATFORM_ID:Linux>:stdc++fs>
  $<$<PLATFORM_ID:Linux>:m>
  roc::rocsolver
)

# Turn on f16c intrinsics
target_compile_options(rocsolver-test PRIVATE -mf16c)
target_compile_definitions(rocsolver-test PRIVATE
  ROCM_USE_FLOAT16
  ROCSOLVER_CLIENTS_TEST
)

add_test(
  NAME rocsolver-test
  COMMAND rocsolver-test
)

rocm_install(TARGETS rocsolver-test COMPONENT tests)
