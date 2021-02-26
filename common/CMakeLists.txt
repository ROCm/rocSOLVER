add_library( rocsolver-common INTERFACE )

target_include_directories( rocsolver-common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set( source_files
  common_host_helpers.cpp
  rocsolver_ostream.cpp
)
prepend_path( "${CMAKE_CURRENT_SOURCE_DIR}/src/" source_files source_paths )
target_sources( rocsolver-common INTERFACE ${source_paths} )
