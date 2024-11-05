/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#include "rocsparselapack_csrlsvqr.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   const rocsparse_mat_descr descA,
                                   const float* A,
                                   const int* ptrA,
                                   const int* indA,
                                   const float* b,
                                   const float tol,
                                   const rocblas_int reorder,
                                   float* x,
                                   rocblas_int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<float>(handle, m, nnz, descA, A, ptrA, indA, b, tol,
                                                     reorder, x, singularity);
}

rocblas_status rocsolver_dcsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   const rocsparse_mat_descr descA,
                                   const double* A,
                                   const rocblas_int* ptrA,
                                   const rocblas_int* indA,
                                   const double* b,
                                   const double tol,
                                   const rocblas_int reorder,
                                   double* x,
                                   rocblas_int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<double>(handle, m, nnz, descA, A, ptrA, indA, b, tol,
                                                      reorder, x, singularity);
}

rocblas_status rocsolver_ccsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   const rocsparse_mat_descr descA,
                                   const rocblas_float_complex* A,
                                   const rocblas_int* ptrA,
                                   const rocblas_int* indA,
                                   const rocblas_float_complex* b,
                                   const rocblas_float_complex tol,
                                   const rocblas_int reorder,
                                   rocblas_float_complex* x,
                                   rocblas_int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_float_complex>(
        handle, m, nnz, descA, A, ptrA, indA, b, tol, reorder, x, singularity);
}

rocblas_status rocsolver_zcsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   const rocsparse_mat_descr descA,
                                   const rocblas_double_complex* A,
                                   const rocblas_int* ptrA,
                                   const rocblas_int* indA,
                                   const rocblas_double_complex* b,
                                   const rocblas_double_complex tol,
                                   const rocblas_int reorder,
                                   rocblas_double_complex* x,
                                   int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_double_complex>(
        handle, m, nnz, descA, A, ptrA, indA, b, tol, reorder, x, singularity);
}

} // extern "C"
