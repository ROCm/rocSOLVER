# ########################################################################
#   Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
#  
#   Redistribution and use in source and binary forms, with or without modification, 
#   are permitted provided that the following conditions are met:
#   1)Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
#   2)Redistributions in binary form must reproduce the above copyright notice, 
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution
# ########################################################################

add_executable(rocsolver-bench client.cpp)

add_armor_flags(rocsolver-bench "${ARMOR_LEVEL}")

target_link_libraries(rocsolver-bench PRIVATE
  Threads::Threads
  hip::device
  rocsolver-common
  clients-common
  roc::rocsolver
)

# Turn on f16c intrinsics
target_compile_options(rocsolver-bench PRIVATE -mf16c)
target_compile_definitions(rocsolver-bench PRIVATE
  ROCM_USE_FLOAT16
  ROCSOLVER_CLIENTS_BENCH
)

rocm_install(TARGETS rocsolver-bench COMPONENT benchmarks)
