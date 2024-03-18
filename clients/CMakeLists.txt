# ##########################################################################
# Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

project(rocsolver-clients LANGUAGES C CXX)
if(UNIX)
  enable_language(Fortran)
endif()

# Specify where to put the client binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# The rocsolver target will exist if the library is being built along with the clients,
# but if this is a clients-only build, we'll have to search for it.
if(NOT TARGET rocsolver)
  find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver)
  get_imported_target_location(location roc::rocsolver)
  message(STATUS "Found rocSOLVER: ${location}")
endif()

if(BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if(ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
    find_package(LAPACK 3.7 REQUIRED CONFIG)
  else()
    find_package(LAPACK 3.7 REQUIRED)
  endif()

  if(NOT LAPACK_LIBRARIES)
    set(LAPACK_LIBRARIES
      ${LAPACK_blas_LIBRARIES}
      ${LAPACK_lapack_LIBRARIES}
    )
  endif()

  add_library(clients-common INTERFACE)
  target_include_directories(clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(clients-common INTERFACE
    ${LAPACK_LIBRARIES}
    $<$<PLATFORM_ID:Linux>:stdc++fs>
    fmt::fmt
  )
  target_link_options(clients-common INTERFACE
    ${LAPACK_LINKER_FLAGS}
  )

  set(rocauxiliary_inst_files
    testcode/auxiliary/testing_lacgv.cpp
    testcode/auxiliary/testing_laswp.cpp
    testcode/auxiliary/testing_larfg.cpp
    testcode/auxiliary/testing_larf.cpp
    testcode/auxiliary/testing_larft.cpp
    testcode/auxiliary/testing_larfb.cpp
    testcode/auxiliary/testing_latrd.cpp
    testcode/auxiliary/testing_labrd.cpp
    testcode/auxiliary/testing_lauum.cpp
    testcode/auxiliary/testing_bdsqr.cpp
    testcode/auxiliary/testing_bdsvdx.cpp
    testcode/auxiliary/testing_steqr.cpp
    testcode/auxiliary/testing_stedc.cpp
    testcode/auxiliary/testing_stedcj.cpp
    testcode/auxiliary/testing_stein.cpp
    testcode/auxiliary/testing_lasyf.cpp
    testcode/auxiliary/testing_sterf.cpp
    testcode/auxiliary/testing_stebz.cpp
    testcode/auxiliary/testing_orgxr_ungxr.cpp
    testcode/auxiliary/testing_orgxl_ungxl.cpp
    testcode/auxiliary/testing_orglx_unglx.cpp
    testcode/auxiliary/testing_orgbr_ungbr.cpp
    testcode/auxiliary/testing_orgtr_ungtr.cpp
    testcode/auxiliary/testing_ormxr_unmxr.cpp
    testcode/auxiliary/testing_ormxl_unmxl.cpp
    testcode/auxiliary/testing_ormlx_unmlx.cpp
    testcode/auxiliary/testing_ormbr_unmbr.cpp
    testcode/auxiliary/testing_ormtr_unmtr.cpp
  )

  set(roclapack_inst_files
    testcode/lapack/testing_potf2_potrf.cpp
    testcode/lapack/testing_potrs.cpp
    testcode/lapack/testing_posv.cpp
    testcode/lapack/testing_potri.cpp
    testcode/lapack/testing_getf2_getrf_npvt.cpp
    testcode/lapack/testing_getf2_getrf.cpp
    testcode/lapack/testing_geqr2_geqrf.cpp
    testcode/lapack/testing_gerq2_gerqf.cpp
    testcode/lapack/testing_geql2_geqlf.cpp
    testcode/lapack/testing_gelq2_gelqf.cpp
    testcode/lapack/testing_getrs.cpp
    testcode/lapack/testing_gesv.cpp
    testcode/lapack/testing_gesvd.cpp
    testcode/lapack/testing_gesvdj.cpp
    testcode/lapack/testing_gesvdx.cpp
    testcode/lapack/testing_trtri.cpp
    testcode/lapack/testing_getri.cpp
    testcode/lapack/testing_getri_npvt.cpp
    testcode/lapack/testing_getri_outofplace.cpp
    testcode/lapack/testing_getri_npvt_outofplace.cpp
    testcode/lapack/testing_gels.cpp
    testcode/lapack/testing_gebd2_gebrd.cpp
    testcode/lapack/testing_sytf2_sytrf.cpp
    testcode/lapack/testing_sytxx_hetxx.cpp
    testcode/lapack/testing_sygsx_hegsx.cpp
    testcode/lapack/testing_syev_heev.cpp
    testcode/lapack/testing_syevd_heevd.cpp
    testcode/lapack/testing_syevdj_heevdj.cpp
    testcode/lapack/testing_syevj_heevj.cpp
    testcode/lapack/testing_syevx_heevx.cpp
    testcode/lapack/testing_sygv_hegv.cpp
    testcode/lapack/testing_sygvd_hegvd.cpp
    testcode/lapack/testing_sygvdj_hegvdj.cpp
    testcode/lapack/testing_sygvj_hegvj.cpp
    testcode/lapack/testing_sygvx_hegvx.cpp
    testcode/lapack/testing_geblttrf_npvt.cpp
    testcode/lapack/testing_geblttrf_npvt_interleaved.cpp
    testcode/lapack/testing_geblttrs_npvt.cpp
    testcode/lapack/testing_geblttrs_npvt_interleaved.cpp
  )

  set(rocrefact_inst_files
    testcode/refact/testing_csrrf_analysis.cpp
    testcode/refact/testing_csrrf_sumlu.cpp
    testcode/refact/testing_csrrf_splitlu.cpp
    testcode/refact/testing_csrrf_refactlu.cpp
    testcode/refact/testing_csrrf_refactchol.cpp
    testcode/refact/testing_csrrf_solve.cpp
  )

  set(common_source_files
    common/lapack_host_reference.cpp
    common/rocsolver_test.cpp
    common/clients_utility.cpp
    common/program_options.cpp
    ${rocauxiliary_inst_files}
    ${roclapack_inst_files}
    ${rocrefact_inst_files}
  )

  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)
  target_sources(clients-common INTERFACE ${common_source_paths})

  # Copy and point to sparse test data
  file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${PROJECT_BINARY_DIR}/staging/sparsedata/
  )
  install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/rocsolver/test
    COMPONENT tests
  )

  target_compile_definitions(clients-common INTERFACE
    -DROCBLAS_NO_DEPRECATED_WARNINGS
  )

  if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()

  if(BUILD_CLIENTS_TESTS)
    add_subdirectory(gtest)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_EXTRA_TESTS)
  add_subdirectory(extras)
endif()
