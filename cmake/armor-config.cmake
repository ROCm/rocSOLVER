# ########################################################################
#   Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
#  
#   Redistribution and use in source and binary forms, with or without modification, 
#   are permitted provided that the following conditions are met:
#   1)Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
#   2)Redistributions in binary form must reproduce the above copyright notice, 
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.
# ########################################################################

# Enables increasingly expensive runtime correctness checks
# 0 - Nothing
# 1 - Inexpensive correctness checks (extra assertions, etc..)
#     Note: Some checks are added by the optimizer, so it can help to build
#           with optimizations enabled. e.g. -Og
# 2 - Expensive correctness checks (debug iterators)
macro(add_armor_flags target level)
  if(UNIX AND "${level}" GREATER "0")
    if("${level}" GREATER "1")
      # Building with std debug iterators is enabled by the defines below, but
      # requires building C++ dependencies with the same defines.
      target_compile_definitions(${target} PRIVATE
        _GLIBCXX_DEBUG
      )
    endif()
    # Note that _FORTIFY_SOURCE does not work unless optimizations are enabled
    target_compile_definitions(${target} PRIVATE
      $<$<NOT:$<BOOL:${BUILD_ADDRESS_SANITIZER}>>:_FORTIFY_SOURCE=1>
      _GLIBCXX_ASSERTIONS
      ROCSOLVER_VERIFY_ASSUMPTIONS
    )
  endif()
endmacro()
