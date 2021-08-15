# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
# ########################################################################

# A helper function to prefix a source list of files with a common path
# into a new list (non-destructive)
function(prepend_path prefix source_list_of_files return_list_of_files)
  foreach(file ${${source_list_of_files}})
    if(IS_ABSOLUTE ${file})
      list(APPEND new_list ${file})
    else()
      list(APPEND new_list ${prefix}/${file})
    endif()
  endforeach()
  set(${return_list_of_files} ${new_list} PARENT_SCOPE)
endfunction()

# Search through the location properties to find one that is set
function(get_imported_target_location result_variable imported_target)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" config)
  set(properties_to_search
    "IMPORTED_LOCATION"
    "IMPORTED_LOCATION_${config}"
    "IMPORTED_LOCATION_DEBUG"
    "IMPORTED_LOCATION_RELEASE"
    "IMPORTED_LOCATION_RELWITHDEBINFO"
    "IMPORTED_LOCATION_MINSIZEREL"
  )
  foreach(property ${properties_to_search})
    get_target_property(location "${imported_target}" "${property}")
    if(location)
      set("${result_variable}" "${location}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set("${result_variable}" "${result_variable}-NOTFOUND" PARENT_SCOPE)
endfunction()

# Define an opposite way of specifying an existing option.
# This may be useful for compatibility.
macro(option_opposite option opposite)
  if(DEFINED "${opposite}")
    if(${opposite})
      set("${option}" OFF)
    else()
      set("${option}" ON)
    endif()
  endif()
endmacro()
