# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

# package_targets is used as a list of install target
set( package_targets rocsolver )

set( rocsolver_lapack_source
  # linear systems solvers
  lapack/roclapack_getrs.cpp
  lapack/roclapack_getrs_batched.cpp
  lapack/roclapack_getrs_strided_batched.cpp
  lapack/roclapack_getri.cpp
  lapack/roclapack_getri_batched.cpp
  lapack/roclapack_getri_strided_batched.cpp
  lapack/roclapack_getri_outofplace.cpp
  lapack/roclapack_getri_outofplace_batched.cpp
  lapack/roclapack_getri_outofplace_strided_batched.cpp
  lapack/roclapack_trtri.cpp
  lapack/roclapack_trtri_batched.cpp
  lapack/roclapack_trtri_strided_batched.cpp
  # least squares solvers
  lapack/roclapack_gels.cpp
  lapack/roclapack_gels_batched.cpp
  lapack/roclapack_gels_strided_batched.cpp
  # triangular factorizations
  lapack/roclapack_getf2.cpp
  lapack/roclapack_getf2_batched.cpp
  lapack/roclapack_getf2_strided_batched.cpp
  lapack/roclapack_getrf.cpp
  lapack/roclapack_getrf_batched.cpp
  lapack/roclapack_getrf_strided_batched.cpp
  lapack/roclapack_potf2.cpp
  lapack/roclapack_potf2_batched.cpp
  lapack/roclapack_potf2_strided_batched.cpp
  lapack/roclapack_potrf.cpp
  lapack/roclapack_potrf_batched.cpp
  lapack/roclapack_potrf_strided_batched.cpp
  # orthogonal factorizations
  lapack/roclapack_geqr2.cpp
  lapack/roclapack_geqr2_batched.cpp
  lapack/roclapack_geqr2_strided_batched.cpp
  lapack/roclapack_geql2.cpp
  lapack/roclapack_geql2_batched.cpp
  lapack/roclapack_geql2_strided_batched.cpp
  lapack/roclapack_gelq2.cpp
  lapack/roclapack_gelq2_batched.cpp
  lapack/roclapack_gelq2_strided_batched.cpp
  lapack/roclapack_geqrf.cpp
  lapack/roclapack_geqrf_batched.cpp
  lapack/roclapack_geqrf_ptr_batched.cpp
  lapack/roclapack_geqrf_strided_batched.cpp
  lapack/roclapack_geqlf.cpp
  lapack/roclapack_geqlf_batched.cpp
  lapack/roclapack_geqlf_strided_batched.cpp
  lapack/roclapack_gelqf.cpp
  lapack/roclapack_gelqf_batched.cpp
  lapack/roclapack_gelqf_strided_batched.cpp
  # Problem and matrix reductions (diagonalizations)
  lapack/roclapack_gebd2.cpp
  lapack/roclapack_gebd2_batched.cpp
  lapack/roclapack_gebd2_strided_batched.cpp
  lapack/roclapack_gebrd.cpp
  lapack/roclapack_gebrd_batched.cpp
  lapack/roclapack_gebrd_strided_batched.cpp
  lapack/roclapack_sytd2_hetd2.cpp
  lapack/roclapack_sytd2_hetd2_batched.cpp
  lapack/roclapack_sytd2_hetd2_strided_batched.cpp
  lapack/roclapack_sytrd_hetrd.cpp
  lapack/roclapack_sytrd_hetrd_batched.cpp
  lapack/roclapack_sytrd_hetrd_strided_batched.cpp
  lapack/roclapack_sygs2_hegs2.cpp
  lapack/roclapack_sygs2_hegs2_batched.cpp
  lapack/roclapack_sygs2_hegs2_strided_batched.cpp
  lapack/roclapack_sygst_hegst.cpp
  lapack/roclapack_sygst_hegst_batched.cpp
  lapack/roclapack_sygst_hegst_strided_batched.cpp
  # singular value decomposition
  lapack/roclapack_gesvd.cpp
  lapack/roclapack_gesvd_batched.cpp
  lapack/roclapack_gesvd_strided_batched.cpp
  # symmetric eigensolvers
  lapack/roclapack_syev_heev.cpp
  lapack/roclapack_syev_heev_batched.cpp
  lapack/roclapack_syev_heev_strided_batched.cpp
  lapack/roclapack_sygv_hegv.cpp
  lapack/roclapack_sygv_hegv_batched.cpp
  lapack/roclapack_sygv_hegv_strided_batched.cpp
)

set( rocsolver_auxiliary_source
  # vector & matrix manipulations
  auxiliary/rocauxiliary_aliases.cpp
  auxiliary/rocauxiliary_lacgv.cpp
  auxiliary/rocauxiliary_laswp.cpp
  # householder reflections
  auxiliary/rocauxiliary_larfg.cpp
  auxiliary/rocauxiliary_larf.cpp
  auxiliary/rocauxiliary_larft.cpp
  auxiliary/rocauxiliary_larfb.cpp
  # orthonormal/unitary matrices
  auxiliary/rocauxiliary_org2r_ung2r.cpp
  auxiliary/rocauxiliary_orgqr_ungqr.cpp
  auxiliary/rocauxiliary_orgl2_ungl2.cpp
  auxiliary/rocauxiliary_orglq_unglq.cpp
  auxiliary/rocauxiliary_org2l_ung2l.cpp
  auxiliary/rocauxiliary_orgql_ungql.cpp
  auxiliary/rocauxiliary_orgbr_ungbr.cpp
  auxiliary/rocauxiliary_orgtr_ungtr.cpp
  auxiliary/rocauxiliary_orm2r_unm2r.cpp
  auxiliary/rocauxiliary_ormqr_unmqr.cpp
  auxiliary/rocauxiliary_orml2_unml2.cpp
  auxiliary/rocauxiliary_ormlq_unmlq.cpp
  auxiliary/rocauxiliary_orm2l_unm2l.cpp
  auxiliary/rocauxiliary_ormql_unmql.cpp
  auxiliary/rocauxiliary_ormbr_unmbr.cpp
  auxiliary/rocauxiliary_ormtr_unmtr.cpp
  # bidiagonal matrices
  auxiliary/rocauxiliary_bdsqr.cpp
  auxiliary/rocauxiliary_labrd.cpp
  # tridiagonal matrices
  auxiliary/rocauxiliary_sterf.cpp
  auxiliary/rocauxiliary_steqr.cpp
  auxiliary/rocauxiliary_stedc.cpp
  auxiliary/rocauxiliary_latrd.cpp
)

set( auxiliaries
  common/buildinfo.cpp
  common/rocsolver_logger.cpp
)

prepend_path( ".." rocsolver_headers_public relative_rocsolver_headers_public )

add_library( rocsolver
  ${relative_rocsolver_headers_public}
  ${auxiliaries}
  ${rocsolver_auxiliary_source}
  ${rocsolver_lapack_source}
)

add_library( roc::rocsolver ALIAS rocsolver )

target_link_libraries( rocsolver
  PUBLIC
    roc::rocblas
  PRIVATE
    $<BUILD_INTERFACE:rocsolver-common> # https://gitlab.kitware.com/cmake/cmake/-/issues/15415
    hip::device
    --rtlib=compiler-rt
    --unwindlib=libgcc
)

# In ROCm 4.0 and earlier, the default maximum threads per block is 256
target_compile_options( rocsolver PRIVATE --gpu-max-threads-per-block=1024 )

# Ignore loop unrolling failures
target_compile_options( rocsolver PRIVATE -Wno-pass-failed )

target_include_directories( rocsolver
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}
          )

set_target_properties( rocsolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
rocm_set_soversion(rocsolver ${rocsolver_SOVERSION})

# set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( rocsolver PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( rocsolver EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsolver-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocsolver PROPERTIES PREFIX "lib" )
endif( )

if(OPTIMAL)
  target_compile_definitions(rocsolver PRIVATE OPTIMAL)
endif( )

target_compile_definitions( rocsolver PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API )

add_armor_flags( rocsolver "${ARMOR_LEVEL}" )

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocsolver
)

rocm_export_targets(
  TARGETS roc::rocsolver
  PREFIX rocsolver
  DEPENDS PACKAGE hip
          PACKAGE rocblas
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( rocsolver )
