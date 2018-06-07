/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "arg_check.h"
#include "rocblas.h"
#include <iostream>

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                             \
  {                                                                            \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                  \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) {                                  \
      fprintf(stderr, "hip error code: %d at %s:%d\n", TMP_STATUS_FOR_CHECK,   \
              __FILE__, __LINE__);                                             \
    }                                                                          \
  }

/* ========================================Gtest Arg Check
 * ===================================================== */

/*! \brief Template: checks if arguments are valid
 */

void potf2_arg_check(rocblas_status status, rocblas_int N) {
#ifdef GOOGLE_TEST
  if (N < 0) {
    ASSERT_EQ(status, rocblas_status_invalid_size);
  } else {
    ASSERT_EQ(status, rocblas_status_success);
  }
#else
  if (N < 0) {
    if (status != rocblas_status_invalid_size)
      std::cerr << "result should be invalid size for size " << N << std::endl;
  } else {
    if (status != rocblas_status_success)
      std::cerr << "result should be success for size " << N << std::endl;
  }
#endif
}

void getf2_arg_check(rocblas_status status, rocblas_int M, rocblas_int N) {
#ifdef GOOGLE_TEST
  if (M < 0 || N < 0) {
    ASSERT_EQ(status, rocblas_status_invalid_size);
  } else {
    ASSERT_EQ(status, rocblas_status_success);
  }
#else
  if (M < 0 || N < 0) {
    if (status != rocblas_status_invalid_size)
      std::cerr << "result should be invalid size for size " << M << " and "
                << N << std::endl;
  } else {
    if (status != rocblas_status_success)
      std::cerr << "result should be success for size " << M << " and " << N
                << std::endl;
  }
#endif
}

void verify_rocblas_status_invalid_pointer(rocblas_status status,
                                           const char *message) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(status, rocblas_status_invalid_pointer);
#else
  if (status != rocblas_status_invalid_pointer) {
    std::cerr << message << std::endl;
  }
#endif
}

void verify_rocblas_status_invalid_size(rocblas_status status,
                                        const char *message) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(status, rocblas_status_invalid_size);
#else
  if (status != rocblas_status_invalid_size) {
    std::cerr << "rocBLAS TEST ERROR: status != rocblas_status_invalid_size, ";
    std::cerr << message << std::endl;
  }
#endif
}

// void handle_check(rocblas_status status)
void verify_rocblas_status_invalid_handle(rocblas_status status) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(status, rocblas_status_invalid_handle);
#else
  if (status != rocblas_status_invalid_handle) {
    std::cerr << "rocBLAS TEST ERROR: handle is null pointer" << std::endl;
  }
#endif
}

void verify_rocblas_status_success(rocblas_status status, const char *message) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(status, rocblas_status_success);
#else
  if (status != rocblas_status_success) {
    std::cerr << message << std::endl;
    std::cerr << "rocBLAS TEST ERROR: status should be rocblas_status_success"
              << std::endl;
    std::cerr << "rocBLAS TEST ERROR: status = " << status << std::endl;
  }
#endif
}

template <> void verify_not_nan(rocblas_half arg) {
// check against 16 bit IEEE NaN immediate value, will work on machine without
// 16 bit IEEE
#ifdef GOOGLE_TEST
  ASSERT_TRUE(static_cast<uint16_t>(arg & 0X7ff) <=
              static_cast<uint16_t>(0X7C00));
#else
  if (static_cast<uint16_t>(arg & 0X7ff) > static_cast<uint16_t>(0X7C00)) {
    std::cerr << "rocBLAS TEST ERROR: argument is NaN" << std::endl;
  }
#endif
}

template <> void verify_not_nan(float arg) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(arg, arg);
#else
  if (arg != arg) {
    std::cerr << "rocBLAS TEST ERROR: argument is NaN" << std::endl;
  }
#endif
}

template <> void verify_not_nan(double arg) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(arg, arg);
#else
  if (arg != arg) {
    std::cerr << "rocBLAS TEST ERROR: argument is NaN" << std::endl;
  }
#endif
}

template <> void verify_equal(int arg1, int arg2, const char *message) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(arg1, arg2);
#else
  if (arg1 != arg2) {
    std::cerr << message << std::endl;
    std::cerr << "rocBLAS TEST ERROR: arguments not equal" << std::endl;
  }
#endif
}

template <> void verify_equal(bool arg1, bool arg2, const char *message) {
#ifdef GOOGLE_TEST
  ASSERT_EQ(arg1, arg2);
#else
  if (arg1 != arg2) {
    std::cerr << message << std::endl;
    std::cerr << "rocBLAS TEST ERROR: arguments not equal" << std::endl;
  }
#endif
}
