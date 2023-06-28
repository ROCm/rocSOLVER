# ########################################################################
#   Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
#  
#   Redistribution and use in source and binary forms, with or without modification, 
#   are permitted provided that the following conditions are met:
#   1)Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
#   2)Redistributions in binary form must reproduce the above copyright notice, 
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.
# ########################################################################

function(get_os_id OS_ID)
  set(_os_id "unknown")
  if(EXISTS "/etc/os-release")
    read_key("ID" _os_id)
  endif()
  if(_os_id STREQUAL "opensuse-leap")
    set(_os_id "sles")
  endif()
  set(${OS_ID} ${_os_id} PARENT_SCOPE)
  set(${OS_ID}_${_os_id} TRUE PARENT_SCOPE)
endfunction()

function(read_key KEYVALUE OUTPUT)
  # Finds the line with the keyvalue
  file(STRINGS /etc/os-release _keyvalue_line REGEX "^${KEYVALUE}=")

  # Remove keyvalue=
  string(REGEX REPLACE "^${KEYVALUE}=\"?(.*)" "\\1" _output "${_keyvalue_line}")

  # Remove trailing quote
  string(REGEX REPLACE "\"$" "" _output "${_output}")
  set(${OUTPUT} ${_output} PARENT_SCOPE)
endfunction()
