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
                                   int m,
                                   int nnz,
                                   const rocsolver_rfinfo rfinfo,
                                   const float* A,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   const float* b,
                                   float tol,
                                   int reorder,
                                   float* x,
                                   int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<float>(handle, m, nnz, rfinfo, A, csrRowPtrA,
                                                     csrColIndA, b, tol, reorder, x, singularity);
}

rocblas_status rocsolver_dcsrlsvqr(rocblas_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsolver_rfinfo rfinfo,
                                   const double* A,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   const double* b,
                                   double tol,
                                   int reorder,
                                   double* x,
                                   int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<double>(handle, m, nnz, rfinfo, A, csrRowPtrA,
                                                      csrColIndA, b, tol, reorder, x, singularity);
}

rocblas_status rocsolver_ccsrlsvqr(rocblas_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsolver_rfinfo rfinfo,
                                   const rocblas_float_complex* A,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   const rocblas_float_complex* b,
                                   rocblas_float_complex tol,
                                   int reorder,
                                   rocblas_float_complex* x,
                                   int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_float_complex>(
        handle, m, nnz, rfinfo, A, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

rocblas_status rocsolver_zcsrlsvqr(rocblas_handle handle,
                                   int m,
                                   int nnz,
                                   const rocsolver_rfinfo rfinfo,
                                   const rocblas_double_complex* A,
                                   const int* csrRowPtrA,
                                   const int* csrColIndA,
                                   const rocblas_double_complex* b,
                                   rocblas_double_complex tol,
                                   int reorder,
                                   rocblas_double_complex* x,
                                   int* singularity)
{
    return rocsolver::rocsolver_csrlsvqr_impl<rocblas_double_complex>(
        handle, m, nnz, rfinfo, A, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

} // extern "C"
