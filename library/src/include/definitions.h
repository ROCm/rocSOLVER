/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

/*******************************************************************************
 * Definitions
 * this file to not include any others
 * thereby it can include top-level definitions included by all
 ******************************************************************************/

// half vectors
typedef __fp16 half8 __attribute__((ext_vector_type(8)));
typedef __fp16 half2 __attribute__((ext_vector_type(2)));
__device__ extern "C" half2 llvm_fma_v2f16(half2, half2, half2) __asm("llvm.fma.v2f16");

__device__ inline half2 rocblas_fmadd_half2(half2 multiplier,
                                            half2 multiplicand, half2 addend) {
  return llvm_fma_v2f16(multiplier, multiplicand, addend);
};

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                            \
  {                                                                            \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                  \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) {                                  \
      return get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK);          \
    }                                                                          \
  }

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                        \
  {                                                                            \
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;              \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) {                      \
      return TMP_STATUS_FOR_CHECK;                                             \
    }                                                                          \
  }

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                             \
  {                                                                            \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                  \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) {                                  \
      throw get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK);           \
    }                                                                          \
  }

#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                         \
  {                                                                            \
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;              \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) {                      \
      throw TMP_STATUS_FOR_CHECK;                                              \
    }                                                                          \
  }

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                             \
  {                                                                            \
    hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                  \
    if (TMP_STATUS_FOR_CHECK != hipSuccess) {                                  \
      fprintf(stderr, "hip error code: %d at %s:%d\n", TMP_STATUS_FOR_CHECK,   \
              __FILE__, __LINE__);                                             \
    }                                                                          \
  }

#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                         \
  {                                                                            \
    rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;              \
    if (TMP_STATUS_FOR_CHECK != rocblas_status_success) {                      \
      fprintf(stderr, "rocblas error code: %d at %s:%d\n",                     \
              TMP_STATUS_FOR_CHECK, __FILE__, __LINE__);                       \
    }                                                                          \
  }

#endif // DEFINITIONS_H
