/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <rocsparse/rocsparse.h>

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
