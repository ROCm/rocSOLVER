# ########################################################################
# Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
# ########################################################################

find_package(GTest REQUIRED)

add_executable(test-header
  test_header.c
)
target_link_libraries(test-header PRIVATE roc::rocsolver)
set_target_properties(test-header PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

if(UNIX AND BUILD_SHARED_LIBS)
  if(TARGET rocsolver)
    add_executable(test-rocsolver-dlopen
      test_dlopen_main.cpp
    )
    target_compile_definitions(test-rocsolver-dlopen PRIVATE
      ROCSOLVER_LIB_NAME="$<TARGET_FILE:rocsolver>"
    )
    target_link_libraries(test-rocsolver-dlopen PRIVATE
      GTest::GTest
      ${CMAKE_DL_LIBS}
    )

    add_test(
      NAME test-rocsolver-dlopen
      COMMAND test-rocsolver-dlopen
    )
  endif()

  if(TARGET rocsolver-bench)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
      add_test(
        NAME test-rocsolver-bench
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_rocsolver_bench.py"
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:rocsolver-bench>"
      )
    endif()
  endif()
endif()
