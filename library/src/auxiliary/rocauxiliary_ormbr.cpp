/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_ormbr.hpp"

template <typename T>
rocblas_status rocsolver_ormbr_impl(rocsolver_handle handle, const rocsolver_storev storev, const rocsolver_side side, const rocsolver_operation trans, 
                                   const rocsolver_int m, const rocsolver_int n, 
                                   const rocsolver_int k, T* A, const rocsolver_int lda, T* ipiv, T *C, const rocsolver_int ldc)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    //logging is missing ???

    if (m < 0 || n < 0 ||  k < 0 || ldc < m)
        return rocblas_status_invalid_size;
    
    rocblas_int nq = side == rocblas_side_left ? m : n;

    if (storev == rocsolver_column_wise && lda < nq)
        return rocblas_status_invalid_size;
    if (storev == rocsolver_row_wise && lda < min(nq,k))
        return rocblas_status_invalid_size;

    if (!A || !ipiv || !C)
        return rocblas_status_invalid_pointer;

    rocblas_int strideA = 0;
    rocblas_int strideP = 0;
    rocblas_int strideC = 0;
    rocblas_int batch_count=1;

    return rocsolver_ormbr_template<false,false,T>(handle,storev,side,trans,
                                                  m,n,k,
                                                  A,0,    //shifted 0 entries
                                                  lda,
                                                  strideA,
                                                  ipiv,
                                                  strideP,
                                                  C,0,  
                                                  ldc,
                                                  strideC,
                                                  batch_count);
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocsolver_status rocsolver_sormbr(rocsolver_handle handle,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_side side,
                                                 const rocsolver_operation trans,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 float *A,
                                                 const rocsolver_int lda,
                                                 float *ipiv,
                                                 float *C,
                                                 const rocsolver_int ldc)
{
    return rocsolver_ormbr_impl<float>(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

ROCSOLVER_EXPORT rocsolver_status rocsolver_dormbr(rocsolver_handle handle,
                                                 const rocsolver_storev storev,
                                                 const rocsolver_side side,
                                                 const rocsolver_operation trans,
                                                 const rocsolver_int m,
                                                 const rocsolver_int n,
                                                 const rocsolver_int k,
                                                 double *A,
                                                 const rocsolver_int lda,
                                                 double *ipiv,
                                                 double *C,
                                                 const rocsolver_int ldc)
{
    return rocsolver_ormbr_impl<double>(handle, storev, side, trans, m, n, k, A, lda, ipiv, C, ldc);
}

} //extern C

