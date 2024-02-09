/* **************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

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

// trsm
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
                        bool inblocked = false,
                        const rocblas_int inca = 1,
                        const rocblas_int incb = 1);

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trsm_lower(rocblas_handle handle,
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
rocblas_status rocsolver_trsm_lower(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int inca,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_int shiftB,
                                    const rocblas_int incb,
                                    const rocblas_int ldb,
                                    const rocblas_stride strideB,
                                    const rocblas_int batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_trsm_upper(rocblas_handle handle,
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
rocblas_status rocsolver_trsm_upper(rocblas_handle handle,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_diagonal diag,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int inca,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    U B,
                                    const rocblas_int shiftB,
                                    const rocblas_int incb,
                                    const rocblas_int ldb,
                                    const rocblas_stride strideB,
                                    const rocblas_int batch_count,
                                    const bool optim_mem,
                                    void* work1,
                                    void* work2,
                                    void* work3,
                                    void* work4);

// gemm
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gemm(rocblas_handle handle,
                              rocblas_operation transA,
                              rocblas_operation transB,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              const T* alpha,
                              U A,
                              rocblas_stride shiftA,
                              rocblas_int lda,
                              rocblas_stride strideA,
                              U B,
                              rocblas_stride shiftB,
                              rocblas_int ldb,
                              rocblas_stride strideB,
                              const T* beta,
                              U C,
                              rocblas_stride shiftC,
                              rocblas_int ldc,
                              rocblas_stride strideC,
                              rocblas_int batch_count,
                              T** work);

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_gemm(rocblas_handle handle,
                              rocblas_operation transA,
                              rocblas_operation transB,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int k,
                              const T* alpha,
                              U A,
                              rocblas_stride shiftA,
                              rocblas_int inca,
                              rocblas_int lda,
                              rocblas_stride strideA,
                              U B,
                              rocblas_stride shiftB,
                              rocblas_int incb,
                              rocblas_int ldb,
                              rocblas_stride strideB,
                              const T* beta,
                              U C,
                              rocblas_stride shiftC,
                              rocblas_int incc,
                              rocblas_int ldc,
                              rocblas_stride strideC,
                              rocblas_int batch_count,
                              T** work);

// ger
template <bool CONJ, typename T, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             rocblas_int m,
                             rocblas_int n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             rocblas_int incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             rocblas_int incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             rocblas_int lda,
                             rocblas_stride strideA,
                             rocblas_int batch_count,
                             T** work);

template <bool CONJ, typename T, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             rocblas_int m,
                             rocblas_int n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             rocblas_int incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             rocblas_int incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             rocblas_int inca,
                             rocblas_int lda,
                             rocblas_stride strideA,
                             rocblas_int batch_count,
                             T** work);

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
