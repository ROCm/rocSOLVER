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
                                   float* A,
                                   int* ptrA,
                                   int* indA,
                                   float* b,
                                   const float tol,
                                   const rocblas_int reorder,
                                   float* x,
                                   rocblas_int* singularity,
                                   rocsolver_spinfo spinfo)
{
    return rocsolver::rocsolver_csrlsvqr_impl<float>(handle, m, nnz, A, ptrA, indA, b, tol, reorder,
                                                     x, singularity, spinfo);
}

rocblas_status rocsolver_dcsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   double* A,
                                   rocblas_int* ptrA,
                                   rocblas_int* indA,
                                   double* b,
                                   const double tol,
                                   const rocblas_int reorder,
                                   double* x,
                                   rocblas_int* singularity,
                                   rocsolver_spinfo spinfo)
{
    return rocsolver::rocsolver_csrlsvqr_impl<double>(handle, m, nnz, A, ptrA, indA, b, tol,
                                                      reorder, x, singularity, spinfo);
}

rocblas_status rocsolver_ccsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   rocblas_float_complex* A,
                                   rocblas_int* ptrA,
                                   rocblas_int* indA,
                                   rocblas_float_complex* b,
                                   const rocblas_float_complex tol,
                                   const rocblas_int reorder,
                                   rocblas_float_complex* x,
                                   rocblas_int* singularity,
                                   rocsolver_spinfo spinfo)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_float_complex>(
        handle, m, nnz, A, ptrA, indA, b, tol, reorder, x, singularity, spinfo);
}

rocblas_status rocsolver_zcsrlsvqr(rocblas_handle handle,
                                   const rocblas_int m,
                                   const rocblas_int nnz,
                                   rocblas_double_complex* A,
                                   rocblas_int* ptrA,
                                   rocblas_int* indA,
                                   rocblas_double_complex* b,
                                   const rocblas_double_complex tol,
                                   const rocblas_int reorder,
                                   rocblas_double_complex* x,
                                   int* singularity,
                                   rocsolver_spinfo spinfo)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_double_complex>(
        handle, m, nnz, A, ptrA, indA, b, tol, reorder, x, singularity, spinfo);
}

} // extern "C"
