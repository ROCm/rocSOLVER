# ########################################################################
# Copyright (c) 2020 Advanced Micro Devices, Inc.
# ########################################################################

# Enables increasingly expensive runtime correctness checks
# 0 - Nothing
# 1 - Inexpensive correctness checks (extra assertions, etc..)
#     Note: Some checks are added by the optimizer, so it can help to build
#           with optimizations enabled. e.g. -Og
# 2 - Expensive correctness checks (debug iterators)
macro( add_armor_flags target level )
  if( "${level}" GREATER "0" )
    if( "${level}" GREATER "1" )
      # Building with std debug iterators is enabled by the defines below, but
      # requires building C++ dependencies with the same defines.
      target_compile_definitions( ${target} PRIVATE
        _GLIBCXX_DEBUG
        _LIBCPP_DEBUG=1
      )
    else( )
      target_compile_definitions( ${target} PRIVATE
        _LIBCPP_DEBUG=0
      )
    endif( )
    target_compile_definitions( ${target} PRIVATE
      _FORTIFY_SOURCE=1 # requires optimizations to work
      _GLIBCXX_ASSERTIONS
      ROCSOLVER_VERIFY_ASSUMPTIONS
    )
  endif( )
endmacro( )
