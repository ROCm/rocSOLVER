/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/*
 * ===========================================================================
 *    common location for declarations of specialized kernel launchers.
 *    Specialized kernels and their launchers are defined in cpp files to
 *    help with compile times.
 * ===========================================================================
 */

template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_trsm_mem(const rocblas_side side,
                        const rocblas_operation trans,
                        const rocblas_int m,
                        const rocblas_int n,
                        const rocblas_int batch_count,
                        size_t* size_work1,
                        size_t* size_work2,
                        size_t* size_work3,
                        size_t* size_work4,
                        bool* optim_mem,
                        bool inblocked = false);

template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_lower(rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_diagonal diag,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4);

template <bool BATCHED, bool STRIDED, typename T, typename U>
void rocsolver_trsm_upper(rocblas_handle handle,
                          const rocblas_side side,
                          const rocblas_operation trans,
                          const rocblas_diagonal diag,
                          const rocblas_int m,
                          const rocblas_int n,
                          U A,
                          const rocblas_int shiftA,
                          const rocblas_int lda,
                          const rocblas_stride strideA,
                          U B,
                          const rocblas_int shiftB,
                          const rocblas_int ldb,
                          const rocblas_stride strideB,
                          const rocblas_int batch_count,
                          const bool optim_mem,
                          void* work1,
                          void* work2,
                          void* work3,
                          void* work4);

#ifdef OPTIMAL

template <typename T, typename U>
rocblas_status getf2_run_panel(rocblas_handle handle,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool pivot,
                               const rocblas_int offset,
                               rocblas_int* permut_idx,
                               const rocblas_stride stride);

template <typename T, typename U>
void getf2_run_scale_update(rocblas_handle handle,
                            const rocblas_int m,
                            const rocblas_int n,
                            T* pivotval,
                            U A,
                            const rocblas_int shiftA,
                            const rocblas_int lda,
                            const rocblas_stride strideA,
                            const rocblas_int batch_count,
                            const rocblas_int dimx,
                            const rocblas_int dimy);

template <typename T, typename U>
rocblas_status getf2_run_small(rocblas_handle handle,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool pivot,
                               const rocblas_int offset,
                               rocblas_int* permut_idx,
                               const rocblas_stride stride);

template <typename T, typename U>
rocblas_status getri_run_small(rocblas_handle handle,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const bool complete,
                               const bool pivot);

template <typename T, typename U>
void trti2_run_small(rocblas_handle handle,
                     const rocblas_fill uplo,
                     const rocblas_diagonal diag,
                     const rocblas_int n,
                     U A,
                     const rocblas_int shiftA,
                     const rocblas_int lda,
                     const rocblas_stride strideA,
                     const rocblas_int batch_count);

#endif // OPTIMAL
