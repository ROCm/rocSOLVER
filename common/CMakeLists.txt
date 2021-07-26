# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
# ########################################################################

add_library(rocsolver-common INTERFACE)

target_include_directories(rocsolver-common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(source_files
  common_host_helpers.cpp
)
prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/src/" source_files source_paths)
target_sources(rocsolver-common INTERFACE ${source_paths})
target_compile_definitions(rocsolver-common INTERFACE __HIP_HCC_COMPAT_MODE__=1)
target_compile_options(rocsolver-common INTERFACE
  -Wno-unused-result # TODO: address [[nodiscard]] warnings
)
if(WIN32)
  target_compile_definitions(rocsolver-common INTERFACE
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    __HIP_ROCclr__=1
    __HIP_PLATFORM_AMD__=1
  )
  target_compile_options(rocsolver-common INTERFACE
    -fms-extensions
    -fms-compatibility
    -Wno-ignored-attributes
    -Wno-unused-command-line-argument
  )
endif()
if(WERROR)
  target_compile_options(rocsolver-common INTERFACE
    -Werror=vla
    -Werror=reorder
    -Werror=return-type
  )
endif()
if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rocsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
  )
  target_link_options(rocsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
    -fuse-ld=lld
  )
endif()
if(BUILD_CODE_COVERAGE)
  target_compile_options(rocsolver-common INTERFACE
    -fprofile-arcs
    -ftest-coverage
  )
  target_link_options(rocsolver-common INTERFACE --coverage)
endif()
