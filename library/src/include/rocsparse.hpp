/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

ROCSOLVER_BEGIN_NAMESPACE

constexpr auto rocsparse2string_status(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success: return "rocsparse_status_success";
    case rocsparse_status_invalid_handle: return "rocsparse_status_invalid_handle";
    case rocsparse_status_not_implemented: return "rocsparse_status_not_implemented";
    case rocsparse_status_invalid_pointer: return "rocsparse_status_invalid_pointer";
    case rocsparse_status_invalid_size: return "rocsparse_status_invalid_size";
    case rocsparse_status_memory_error: return "rocsparse_status_memory_error";
    case rocsparse_status_internal_error: return "rocsparse_status_internal_error";
    case rocsparse_status_invalid_value: return "rocsparse_status_invalid_value";
    case rocsparse_status_arch_mismatch: return "rocsparse_status_arch_mismatch";
    case rocsparse_status_zero_pivot: return "rocsparse_status_zero_pivot";
    case rocsparse_status_not_initialized: return "rocsparse_status_not_initialized";
    case rocsparse_status_type_mismatch: return "rocsparse_status_type_mismatch";
    case rocsparse_status_requires_sorted_storage:
        return "rocsparse_status_requires_sorted_storage";
    default: return "unknown";
    }
}

constexpr auto rocsparse2rocblas_status(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success: return rocblas_status_success;
    case rocsparse_status_invalid_handle: return rocblas_status_invalid_handle;
    case rocsparse_status_not_implemented: return rocblas_status_not_implemented;
    case rocsparse_status_invalid_pointer: return rocblas_status_invalid_pointer;
    case rocsparse_status_invalid_size: return rocblas_status_invalid_size;
    case rocsparse_status_memory_error: return rocblas_status_memory_error;
    case rocsparse_status_invalid_value: return rocblas_status_invalid_value;
    default: return rocblas_status_internal_error;
    }
}

#define ROCSPARSE_CHECK(...)                                     \
    {                                                            \
        rocsparse_status _status = (__VA_ARGS__);                \
        if(_status != rocsparse_status_success)                  \
            return rocsolver::rocsparse2rocblas_status(_status); \
    }
#define THROW_IF_ROCSPARSE_ERROR(...)                           \
    {                                                           \
        rocsparse_status _status = (__VA_ARGS__);               \
        if(_status != rocsparse_status_success)                 \
            throw rocsolver::rocsparse2rocblas_status(_status); \
    }

// csrilu0 buffer
inline rocsparse_status rocsparseCall_csrilu0_buffer_size(rocsparse_handle sphandle,
                                                          rocblas_int n,
                                                          rocblas_int nnz,
                                                          rocsparse_mat_descr descr,
                                                          float* val,
                                                          rocblas_int* ptr,
                                                          rocblas_int* ind,
                                                          rocsparse_mat_info info,
                                                          size_t* size)
{
    return rocsparse_scsrilu0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

inline rocsparse_status rocsparseCall_csrilu0_buffer_size(rocsparse_handle sphandle,
                                                          rocblas_int n,
                                                          rocblas_int nnz,
                                                          rocsparse_mat_descr descr,
                                                          double* val,
                                                          rocblas_int* ptr,
                                                          rocblas_int* ind,
                                                          rocsparse_mat_info info,
                                                          size_t* size)
{
    return rocsparse_dcsrilu0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

// csric0 buffer
inline rocsparse_status rocsparseCall_csric0_buffer_size(rocsparse_handle sphandle,
                                                         rocblas_int n,
                                                         rocblas_int nnz,
                                                         rocsparse_mat_descr descr,
                                                         float* val,
                                                         rocblas_int* ptr,
                                                         rocblas_int* ind,
                                                         rocsparse_mat_info info,
                                                         size_t* size)
{
    return rocsparse_scsric0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

inline rocsparse_status rocsparseCall_csric0_buffer_size(rocsparse_handle sphandle,
                                                         rocblas_int n,
                                                         rocblas_int nnz,
                                                         rocsparse_mat_descr descr,
                                                         double* val,
                                                         rocblas_int* ptr,
                                                         rocblas_int* ind,
                                                         rocsparse_mat_info info,
                                                         size_t* size)
{
    return rocsparse_dcsric0_buffer_size(sphandle, n, nnz, descr, val, ptr, ind, info, size);
}

// csrilu0 analysis
inline rocsparse_status rocsparseCall_csrilu0_analysis(rocsparse_handle sphandle,
                                                       rocblas_int n,
                                                       rocblas_int nnz,
                                                       rocsparse_mat_descr descr,
                                                       float* val,
                                                       rocblas_int* ptr,
                                                       rocblas_int* ind,
                                                       rocsparse_mat_info info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy solve,
                                                       void* buffer)
{
    return rocsparse_scsrilu0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis,
                                       solve, buffer);
}

inline rocsparse_status rocsparseCall_csrilu0_analysis(rocsparse_handle sphandle,
                                                       rocblas_int n,
                                                       rocblas_int nnz,
                                                       rocsparse_mat_descr descr,
                                                       double* val,
                                                       rocblas_int* ptr,
                                                       rocblas_int* ind,
                                                       rocsparse_mat_info info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy solve,
                                                       void* buffer)
{
    return rocsparse_dcsrilu0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis,
                                       solve, buffer);
}

// csric0 analysis
inline rocsparse_status rocsparseCall_csric0_analysis(rocsparse_handle sphandle,
                                                      rocblas_int n,
                                                      rocblas_int nnz,
                                                      rocsparse_mat_descr descr,
                                                      float* val,
                                                      rocblas_int* ptr,
                                                      rocblas_int* ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* buffer)
{
    return rocsparse_scsric0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis, solve,
                                      buffer);
}

inline rocsparse_status rocsparseCall_csric0_analysis(rocsparse_handle sphandle,
                                                      rocblas_int n,
                                                      rocblas_int nnz,
                                                      rocsparse_mat_descr descr,
                                                      double* val,
                                                      rocblas_int* ptr,
                                                      rocblas_int* ind,
                                                      rocsparse_mat_info info,
                                                      rocsparse_analysis_policy analysis,
                                                      rocsparse_solve_policy solve,
                                                      void* buffer)
{
    return rocsparse_dcsric0_analysis(sphandle, n, nnz, descr, val, ptr, ind, info, analysis, solve,
                                      buffer);
}

// csrilu0
inline rocsparse_status rocsparseCall_csrilu0(rocsparse_handle sphandle,
                                              rocblas_int n,
                                              rocblas_int nnz,
                                              rocsparse_mat_descr descr,
                                              float* val,
                                              rocblas_int* ptr,
                                              rocblas_int* ind,
                                              rocsparse_mat_info info,
                                              rocsparse_solve_policy solve,
                                              void* buffer)
{
    return rocsparse_scsrilu0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrilu0(rocsparse_handle sphandle,
                                              rocblas_int n,
                                              rocblas_int nnz,
                                              rocsparse_mat_descr descr,
                                              double* val,
                                              rocblas_int* ptr,
                                              rocblas_int* ind,
                                              rocsparse_mat_info info,
                                              rocsparse_solve_policy solve,
                                              void* buffer)
{
    return rocsparse_dcsrilu0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

// csric0
inline rocsparse_status rocsparseCall_csric0(rocsparse_handle sphandle,
                                             rocblas_int n,
                                             rocblas_int nnz,
                                             rocsparse_mat_descr descr,
                                             float* val,
                                             rocblas_int* ptr,
                                             rocblas_int* ind,
                                             rocsparse_mat_info info,
                                             rocsparse_solve_policy solve,
                                             void* buffer)
{
    return rocsparse_scsric0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csric0(rocsparse_handle sphandle,
                                             rocblas_int n,
                                             rocblas_int nnz,
                                             rocsparse_mat_descr descr,
                                             double* val,
                                             rocblas_int* ptr,
                                             rocblas_int* ind,
                                             rocsparse_mat_info info,
                                             rocsparse_solve_policy solve,
                                             void* buffer)
{
    return rocsparse_dcsric0(sphandle, n, nnz, descr, val, ptr, ind, info, solve, buffer);
}

// csrsm buffer
inline rocsparse_status rocsparseCall_csrsm_buffer_size(rocsparse_handle sphandle,
                                                        rocsparse_operation transA,
                                                        rocsparse_operation transB,
                                                        rocblas_int n,
                                                        rocblas_int nrhs,
                                                        rocblas_int nnz,
                                                        float* alpha,
                                                        rocsparse_mat_descr descr,
                                                        float* val,
                                                        rocblas_int* ptr,
                                                        rocblas_int* ind,
                                                        float* B,
                                                        rocblas_int ldb,
                                                        rocsparse_mat_info info,
                                                        rocsparse_solve_policy solve,
                                                        size_t* size)
{
    return rocsparse_scsrsm_buffer_size(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val,
                                        ptr, ind, B, ldb, info, solve, size);
}

inline rocsparse_status rocsparseCall_csrsm_buffer_size(rocsparse_handle sphandle,
                                                        rocsparse_operation transA,
                                                        rocsparse_operation transB,
                                                        rocblas_int n,
                                                        rocblas_int nrhs,
                                                        rocblas_int nnz,
                                                        double* alpha,
                                                        rocsparse_mat_descr descr,
                                                        double* val,
                                                        rocblas_int* ptr,
                                                        rocblas_int* ind,
                                                        double* B,
                                                        rocblas_int ldb,
                                                        rocsparse_mat_info info,
                                                        rocsparse_solve_policy solve,
                                                        size_t* size)
{
    return rocsparse_dcsrsm_buffer_size(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val,
                                        ptr, ind, B, ldb, info, solve, size);
}

// csrsm analysis
inline rocsparse_status rocsparseCall_csrsm_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation transA,
                                                     rocsparse_operation transB,
                                                     rocblas_int n,
                                                     rocblas_int nrhs,
                                                     rocblas_int nnz,
                                                     float* alpha,
                                                     rocsparse_mat_descr descr,
                                                     float* val,
                                                     rocblas_int* ptr,
                                                     rocblas_int* ind,
                                                     float* B,
                                                     rocblas_int ldb,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
                                                     void* buffer)
{
    return rocsparse_scsrsm_analysis(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                     ind, B, ldb, info, analysis, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrsm_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation transA,
                                                     rocsparse_operation transB,
                                                     rocblas_int n,
                                                     rocblas_int nrhs,
                                                     rocblas_int nnz,
                                                     double* alpha,
                                                     rocsparse_mat_descr descr,
                                                     double* val,
                                                     rocblas_int* ptr,
                                                     rocblas_int* ind,
                                                     double* B,
                                                     rocblas_int ldb,
                                                     rocsparse_mat_info info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy solve,
                                                     void* buffer)
{
    return rocsparse_dcsrsm_analysis(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                     ind, B, ldb, info, analysis, solve, buffer);
}

//csrsm solve
inline rocsparse_status rocsparseCall_csrsm_solve(rocsparse_handle sphandle,
                                                  rocsparse_operation transA,
                                                  rocsparse_operation transB,
                                                  rocblas_int n,
                                                  rocblas_int nrhs,
                                                  rocblas_int nnz,
                                                  float* alpha,
                                                  rocsparse_mat_descr descr,
                                                  float* val,
                                                  rocblas_int* ptr,
                                                  rocblas_int* ind,
                                                  float* B,
                                                  rocblas_int ldb,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy solve,
                                                  void* buffer)
{
    return rocsparse_scsrsm_solve(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                  ind, B, ldb, info, solve, buffer);
}

inline rocsparse_status rocsparseCall_csrsm_solve(rocsparse_handle sphandle,
                                                  rocsparse_operation transA,
                                                  rocsparse_operation transB,
                                                  rocblas_int n,
                                                  rocblas_int nrhs,
                                                  rocblas_int nnz,
                                                  double* alpha,
                                                  rocsparse_mat_descr descr,
                                                  double* val,
                                                  rocblas_int* ptr,
                                                  rocblas_int* ind,
                                                  double* B,
                                                  rocblas_int ldb,
                                                  rocsparse_mat_info info,
                                                  rocsparse_solve_policy solve,
                                                  void* buffer)
{
    return rocsparse_dcsrsm_solve(sphandle, transA, transB, n, nrhs, nnz, alpha, descr, val, ptr,
                                  ind, B, ldb, info, solve, buffer);
}

ROCSOLVER_END_NAMESPACE
