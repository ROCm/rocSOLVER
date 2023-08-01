# ########################################################################
# Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
# ########################################################################

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
  common/testing_lacgv.cpp
    common/testing_laswp.cpp
    common/testing_larfg.cpp
    common/testing_larf.cpp
    common/testing_larft.cpp
    common/testing_larfb.cpp
    common/testing_latrd.cpp
    common/testing_labrd.cpp
    common/testing_lauum.cpp
    common/testing_bdsqr.cpp
    common/testing_bdsvdx.cpp
    common/testing_steqr.cpp
    common/testing_stedc.cpp
    common/testing_stein.cpp
    common/testing_lasyf.cpp
    common/testing_sterf.cpp
    common/testing_stebz.cpp
    common/testing_orgxr_ungxr.cpp
    common/testing_orgxl_ungxl.cpp
    common/testing_orglx_unglx.cpp
    common/testing_orgbr_ungbr.cpp
    common/testing_orgtr_ungtr.cpp
    common/testing_ormxr_unmxr.cpp
    common/testing_ormxl_unmxl.cpp
    common/testing_ormlx_unmlx.cpp
    common/testing_ormbr_unmbr.cpp
    common/testing_ormtr_unmtr.cpp
  )

  set(roclapack_inst_files
    common/testing_potf2_potrf.cpp
    common/testing_potrs.cpp
    common/testing_posv.cpp
    common/testing_potri.cpp
    common/testing_getf2_getrf_npvt.cpp
    common/testing_getf2_getrf.cpp
    common/testing_geqr2_geqrf.cpp
    common/testing_gerq2_gerqf.cpp
    common/testing_geql2_geqlf.cpp
    common/testing_gelq2_gelqf.cpp
    common/testing_getrs.cpp
    common/testing_gesv.cpp
    common/testing_gesvd.cpp
    common/testing_gesvdj.cpp
    common/testing_gesvdx.cpp
    common/testing_trtri.cpp
    common/testing_getri.cpp
    common/testing_getri_npvt.cpp
    common/testing_getri_outofplace.cpp
    common/testing_getri_npvt_outofplace.cpp
    common/testing_gels.cpp
    common/testing_gebd2_gebrd.cpp
    common/testing_sytf2_sytrf.cpp
    common/testing_sytxx_hetxx.cpp
    common/testing_sygsx_hegsx.cpp
    common/testing_syev_heev.cpp
    common/testing_syevd_heevd.cpp
    common/testing_syevj_heevj.cpp
    common/testing_syevx_heevx.cpp
    common/testing_sygv_hegv.cpp
    common/testing_sygvd_hegvd.cpp
    common/testing_sygvj_hegvj.cpp
    common/testing_sygvx_hegvx.cpp
    common/testing_geblttrf_npvt.cpp
    common/testing_geblttrf_npvt_interleaved.cpp
    common/testing_geblttrs_npvt.cpp
    common/testing_geblttrs_npvt_interleaved.cpp
  )

  set(rocrefact_inst_files
    common/testing_csrrf_analysis.cpp
    common/testing_csrrf_sumlu.cpp
    common/testing_csrrf_splitlu.cpp
    common/testing_csrrf_refactlu.cpp
    common/testing_csrrf_solve.cpp
  )

  set(common_source_files
    common/lapack_host_reference.cpp
    common/rocsolver_test.cpp
    rocblascommon/clients_utility.cpp
    rocblascommon/program_options.cpp
    ${rocauxiliary_inst_files}
    ${roclapack_inst_files}
    ${rocrefact_inst_files}
  )

  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)
  target_sources(clients-common INTERFACE ${common_source_paths})

  # Copy and point to sparse test data
  file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${PROJECT_BINARY_DIR}/sparsedata/
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
