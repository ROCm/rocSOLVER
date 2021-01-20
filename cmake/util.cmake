# ########################################################################
# HELPER FUNCTIONS
# ########################################################################

# ########################################################################
# target_compile_features() override
# Wraps the normal cmake function to cope with hipcc/nvcc weirdness.
# ########################################################################
function( target_compile_features target_name )
  # With Cmake v3.5, hipcc (with nvcc backend) does not work with target_compile_features
  # Turn on -std=c++14 manually
  if( CUDA_FOUND AND CMAKE_CXX_COMPILER MATCHES ".*/hipcc$|.*/nvcc$" )
    set_target_properties( ${target_name} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON )
  else( )
    _target_compile_features( ${target_name} ${ARGN} )
  endif( )
endfunction( )

# ########################################################################
# target_link_libraries() override
# Wraps the normal cmake function to cope with hipcc/nvcc weirdness.
# ########################################################################
function( target_link_libraries target_name )
  # hipcc takes care of finding hip library dependencies internally; remove
  # explicit mentions of them so cmake doesn't complain on nvcc path
  # FIXME: This removes this target's hip libraries, even if this particular
  #        target is not built using a compiler that can find its dependencies
  #        internally (e.g. Fortran programs).
  if( CUDA_FOUND AND CMAKE_CXX_COMPILER MATCHES ".*/hipcc$|.*/nvcc$" )
    foreach( link_library ${ARGN} )
      if( (link_library MATCHES "^hip::") )
        message( DEBUG "Removing ${link_library} from ${target_name} library list" )
      else( )
        if( TARGET ${link_library} )
          list( APPEND new_list -Xlinker ${link_library} )
        else( )
          list( APPEND new_list ${link_library} )
        endif( )
      endif( )
    endforeach( )
    _target_link_libraries( ${target_name} ${new_list} )
  else( )
    _target_link_libraries( ${target_name} ${ARGN} )
  endif( )
endfunction( )

# ########################################################################
# A helper function to prefix a source list of files with a common path
# into a new list (non-destructive)
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

