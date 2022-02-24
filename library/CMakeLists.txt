# ########################################################################
# Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.1)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver-version.h")

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  include(${PROJECT_SOURCE_DIR}/cmake/rocsolver-backward-compat.cmake)
  package_wrapper_headerfile( "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h" )
  message( STATUS "Completed package_wrapper_headerfile() for rocblas version")
endif( )

# Copy source headers to build dir
configure_file(include/rocsolver.h "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver.h" COPYONLY)
configure_file(include/rocsolver-aliases.h "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-aliases.h" COPYONLY)
configure_file(include/rocsolver-functions.h "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-functions.h" COPYONLY)
configure_file(include/rocsolver-extra-types.h "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-extra-types.h" COPYONLY)


set(rocsolver_headers_public
  include/rocsolver.h
  include/rocsolver-extra-types.h
  include/rocsolver-aliases.h
  include/rocsolver-functions.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-export.h
)

# For IDE integration
source_group("Header Files\\Public" FILES ${rocsolver_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(src)

# Package-specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocblas (>= 2.44), rocblas (<< 2.45)")
set(CPACK_RPM_PACKAGE_REQUIRES "rocblas >= 2.44, rocblas < 2.45")

if(OS_ID_sles)
  set(RPM_REQUIREMENTS "${RPM_REQUIREMENTS}, libLLVM7 >= 7.0.1")
endif()
set(CPACK_RPM_PACKAGE_REQUIRES ${RPM_REQUIREMENTS})

set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "BSD")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
  "\${CPACK_PACKAGING_INSTALL_PREFIX}"
  "\${CPACK_PACKAGING_INSTALL_PREFIX}/include"
  "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib"
)

set(ROCSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME rocsolver
  DESCRIPTION "AMD ROCm SOLVER library"
  MAINTAINER "RocSOLVER maintainer <rocsolver-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCSOLVER_CONFIG_DIR}
)
