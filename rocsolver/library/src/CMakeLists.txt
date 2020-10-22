# ########################################################################
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
# ########################################################################

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################

# package_targets is used as a list of install target
set( package_targets rocsolver )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

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
  # bidiagonal matrices and svd
  auxiliary/rocauxiliary_bdsqr.cpp
  auxiliary/rocauxiliary_labrd.cpp
)

set( rocsolver_lapack_source
  # triangular factorizations and linear solvers
  lapack/roclapack_getf2.cpp
  lapack/roclapack_getf2_batched.cpp
  lapack/roclapack_getf2_strided_batched.cpp
  lapack/roclapack_getrf.cpp
  lapack/roclapack_getrf_batched.cpp
  lapack/roclapack_getrf_strided_batched.cpp
  lapack/roclapack_getrs.cpp
  lapack/roclapack_getrs_batched.cpp
  lapack/roclapack_getrs_strided_batched.cpp
  lapack/roclapack_getri.cpp
  lapack/roclapack_getri_batched.cpp
  lapack/roclapack_getri_strided_batched.cpp
  lapack/roclapack_getri_outofplace_batched.cpp
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
  # bidiagonalization and svd
  lapack/roclapack_gebd2.cpp
  lapack/roclapack_gebd2_batched.cpp
  lapack/roclapack_gebd2_strided_batched.cpp
  lapack/roclapack_gebrd.cpp
  lapack/roclapack_gebrd_batched.cpp
  lapack/roclapack_gebrd_strided_batched.cpp
  lapack/roclapack_gesvd.cpp
  lapack/roclapack_gesvd_batched.cpp
  lapack/roclapack_gesvd_strided_batched.cpp
)

set( auxiliaries
  buildinfo.cpp
)

prepend_path( ".." rocsolver_headers_public relative_rocsolver_headers_public )

add_library( rocsolver
  ${relative_rocsolver_headers_public}
  ${auxiliaries}
  ${rocsolver_auxiliary_source}
  ${rocsolver_lapack_source}
)

add_library( roc::rocsolver ALIAS rocsolver )

target_link_libraries( rocsolver PRIVATE hip::device --rtlib=compiler-rt --unwindlib=libgcc )
set_target_properties( rocsolver PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON )

if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
  # Remove following when hcc is fixed; hcc emits following spurious warning ROCm v1.6.1
  # "clang-5.0: warning: argument unused during compilation: '-isystem /opt/rocm/include'"
  target_compile_options( rocsolver PRIVATE -Wno-unused-command-line-argument -fno-gpu-rdc )
  foreach( target ${AMDGPU_TARGETS} )
    target_compile_options( rocsolver PRIVATE --amdgpu-target=${target} )
  endforeach( )
endif( )

target_include_directories( rocsolver
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/rocsolver/library/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
          )

set_target_properties( rocsolver PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( rocsolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
rocm_set_soversion(rocsolver ${rocsolver_SOVERSION})


# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( rocsolver PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( rocsolver EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsolver-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocsolver PROPERTIES PREFIX "lib" )
endif( )

if(OPTIMAL)
  target_compile_definitions(rocsolver PUBLIC OPTIMAL)
endif( )

add_armor_flags( rocsolver "${ARMOR_LEVEL}" )

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_SOURCE_DIR}/rocsolver/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocsolver
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS roc::rocsolver
  PREFIX rocsolver
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( rocsolver )
