/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver.h"

/*
 * ===========================================================================
 *    common location for declarations of small-size kernel launchers. Small-
 *    size kernels and their launchers are defined in cpp files to help with
 *    compile times.
 * ===========================================================================
 */

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
