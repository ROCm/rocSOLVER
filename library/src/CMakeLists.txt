# ########################################################################
# Copyright 2016-2019 Advanced Micro Devices, Inc.
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
  auxiliary/rocauxiliary_laswp.cpp
  auxiliary/rocauxiliary_larfg.cpp
  auxiliary/rocauxiliary_larf.cpp
  auxiliary/rocauxiliary_larft.cpp
  auxiliary/rocauxiliary_larfb.cpp
  auxiliary/rocauxiliary_org2r.cpp
  auxiliary/rocauxiliary_orgqr.cpp
  auxiliary/rocauxiliary_orgl2.cpp
  auxiliary/rocauxiliary_orglq.cpp
  auxiliary/rocauxiliary_orgbr.cpp
  auxiliary/rocauxiliary_orm2r.cpp
  auxiliary/rocauxiliary_ormqr.cpp
)

set( rocsolver_common_source
  common/helpers.cpp
  common/rocblas.cpp
)

set( rocsolver_lapack_source
  lapack/roclapack_getf2.cpp
  lapack/roclapack_getf2_batched.cpp
  lapack/roclapack_getf2_strided_batched.cpp
  lapack/roclapack_getrf.cpp
  lapack/roclapack_getrf_batched.cpp
  lapack/roclapack_getrf_strided_batched.cpp
  lapack/roclapack_getrs.cpp
  lapack/roclapack_getrs_batched.cpp
  lapack/roclapack_getrs_strided_batched.cpp
  lapack/roclapack_potf2.cpp
  lapack/roclapack_potf2_batched.cpp
  lapack/roclapack_potf2_strided_batched.cpp
  lapack/roclapack_potrf.cpp
  lapack/roclapack_potrf_batched.cpp
  lapack/roclapack_potrf_strided_batched.cpp
  lapack/roclapack_geqr2.cpp
  lapack/roclapack_geqr2_batched.cpp
  lapack/roclapack_geqr2_strided_batched.cpp
  lapack/roclapack_gelq2.cpp
  lapack/roclapack_gelq2_batched.cpp
  lapack/roclapack_gelq2_strided_batched.cpp
  lapack/roclapack_geqrf.cpp
  lapack/roclapack_geqrf_batched.cpp
  lapack/roclapack_geqrf_strided_batched.cpp
  lapack/roclapack_gelqf.cpp
  lapack/roclapack_gelqf_batched.cpp
  lapack/roclapack_gelqf_strided_batched.cpp
)

prepend_path( ".." rocsolver_headers_public relative_rocsolver_headers_public )

add_library( rocsolver
  ${rocsolver_lapack_source}
  ${relative_rocsolver_headers_public}
  ${rocsolver_auxiliary_source}
  ${rocsolver_common_source}  
)

add_library( roc::rocsolver ALIAS rocsolver )

# Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
  target_link_libraries( rocsolver PRIVATE hip::device )
else()
  target_link_libraries( rocsolver PRIVATE hip::device hcc::hccshared )
endif()

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
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<BUILD_INTERFACE:${Tensile_INC}>
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

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX rocsolver
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

rocm_export_targets(
  TARGETS rocsolver-targets
  PREFIX rocsolver
  DEPENDS PACKAGE hip
  NAMESPACE roc::
 )

rocm_install_symlink_subdir( rocsolver )
