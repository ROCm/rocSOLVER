# ########################################################################
# Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# declare sample programs
add_executable(example-c-basic
  example_basic.c
)
add_executable(example-cpp-basic
  example_basic.cpp
)
add_executable(example-c-hmm
  example_hmm.c
)
add_executable(example-cpp-logging
  example_logging.cpp
)
add_executable(example-c-batched
  example_batched.c
)
add_executable(example-c-strided-batched
  example_strided_batched.c
)

# group sample programs by language
set(c_samples
  example-c-basic
  example-c-hmm
  example-c-batched
  example-c-strided-batched
)
set(cpp_samples
  example-cpp-basic
  example-cpp-logging
)

# set flags for building the sample programs
foreach(exe ${c_samples} ${cpp_samples} ${fortran_samples})
  target_link_libraries(${exe} PRIVATE roc::rocsolver)
  rocm_install(TARGETS ${exe} COMPONENT samples)
endforeach()

foreach(exe ${cpp_samples})
  set_target_properties(${exe} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endforeach()

foreach(exe ${c_samples})
  set_target_properties(${exe} PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
endforeach()
