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
    return rocsparse_scsrilu0_buffer_size(sphandle, (rocsparse_int)n, (rocsparse_int)nnz, descr,
                                          val, (rocsparse_int*)ptr, (rocsparse_int*)ind, info, size);
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
    return rocsparse_dcsrilu0_buffer_size(sphandle, (rocsparse_int)n, (rocsparse_int)nnz, descr,
                                          val, (rocsparse_int*)ptr, (rocsparse_int*)ind, info, size);
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
    return rocsparse_scsrilu0_analysis(sphandle, (rocsparse_int)n, (rocsparse_int)nnz, descr, val,
                                       (rocsparse_int*)ptr, (rocsparse_int*)ind, info, analysis,
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
    return rocsparse_dcsrilu0_analysis(sphandle, (rocsparse_int)n, (rocsparse_int)nnz, descr, val,
                                       (rocsparse_int*)ptr, (rocsparse_int*)ind, info, analysis,
                                       solve, buffer);
}

// csrsv analysis
inline rocsparse_status rocsparseCall_csrsv_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation trans,
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
    return rocsparse_scsrsv_analysis(sphandle, trans, (rocsparse_int)n, (rocsparse_int)nnz, descr,
                                     val, (rocsparse_int*)ptr, (rocsparse_int*)ind, info, analysis,
                                     solve, buffer);
}

inline rocsparse_status rocsparseCall_csrsv_analysis(rocsparse_handle sphandle,
                                                     rocsparse_operation trans,
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
    return rocsparse_dcsrsv_analysis(sphandle, trans, (rocsparse_int)n, (rocsparse_int)nnz, descr,
                                     val, (rocsparse_int*)ptr, (rocsparse_int*)ind, info, analysis,
                                     solve, buffer);
}
