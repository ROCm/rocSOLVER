/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocauxiliary_labrd.hpp"

template <typename S, typename T, typename U>
rocblas_status rocsolver_labrd_impl(rocblas_handle handle, const rocblas_int m, const rocblas_int n,
                                    const rocblas_int k, U A, const rocblas_int lda, S* D, S* E, T* tauq, T* taup,
                                    U X, const rocblas_int ldx, U Y, const rocblas_int ldy)
{ 
    if(!handle)
        return rocblas_status_invalid_handle;
    
    //logging is missing ???    
    
    // argument checking
    rocblas_status st = rocsolver_labrd_argCheck(m,n,k,lda,ldx,ldy,A,D,E,tauq,taup,X,Y);
    if (st != rocblas_status_continue)
        return st;

    rocblas_stride strideA = 0;
    rocblas_stride strideX = 0;
    rocblas_stride strideY = 0;
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideQ = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory managment
    size_t size_1;  //size of constants
    size_t size_2;  //size of workspace
    size_t size_3;  //size of array of pointers to workspace
    size_t size_4;  //size of cache for norms
    rocsolver_labrd_getMemorySize<T,false>(m,n,batch_count,&size_1,&size_2,&size_3,&size_4);

    // (TODO) MEMORY SIZE QUERIES AND ALLOCATIONS TO BE DONE WITH ROCBLAS HANDLE
    void *scalars, *work, *workArr, *norms;
    hipMalloc(&scalars,size_1);
    hipMalloc(&work,size_2);
    hipMalloc(&workArr,size_3);
    hipMalloc(&norms,size_4);
    if (!scalars || (size_2 && !work) || (size_3 && !workArr) || (size_4 && !norms))
        return rocblas_status_memory_error;

    // scalar constants for rocblas functions calls
    // (to standarize and enable re-use, size_1 always equals 3*sizeof(T))
    T sca[] = { -1, 0, 1 };
    RETURN_IF_HIP_ERROR(hipMemcpy(scalars, sca, size_1, hipMemcpyHostToDevice));

    // execution
    rocblas_status status =
           rocsolver_labrd_template<S,T>(handle,m,n,k,
                                         A,0,    //the matrix is shifted 0 entries (will work on the entire matrix)
                                         lda,strideA,
                                         D,strideD,
                                         E,strideE,
                                         tauq,strideQ,
                                         taup,strideP,
                                         X,0,
                                         ldx,strideX,
                                         Y,0,
                                         ldy,strideY,
                                         batch_count,
                                         (T*)scalars,
                                         (T*)work,
                                         (T**)workArr,
                                         (T*)norms);

    hipFree(scalars);
    hipFree(work);
    hipFree(workArr);
    hipFree(norms);
    return status;
}


/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_slabrd(rocblas_handle handle, const rocblas_int m, const rocblas_int n, const rocblas_int k,
                 float *A, const rocblas_int lda, float *D, float *E, float *tauq, float *taup, float *X, const rocblas_int ldx,
                 float *Y, const rocblas_int ldy)
{
    return rocsolver_labrd_impl<float,float>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dlabrd(rocblas_handle handle, const rocblas_int m, const rocblas_int n, const rocblas_int k,
                 double *A, const rocblas_int lda, double *D, double *E, double *tauq, double *taup, double *X, const rocblas_int ldx,
                 double *Y, const rocblas_int ldy)
{
    return rocsolver_labrd_impl<double,double>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_clabrd(rocblas_handle handle, const rocblas_int m, const rocblas_int n, const rocblas_int k,
                 rocblas_float_complex *A, const rocblas_int lda, float *D, float *E, rocblas_float_complex *tauq, rocblas_float_complex *taup,
                 rocblas_float_complex *X, const rocblas_int ldx, rocblas_float_complex *Y, const rocblas_int ldy)
{
    return rocsolver_labrd_impl<float,rocblas_float_complex>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zlabrd(rocblas_handle handle, const rocblas_int m, const rocblas_int n, const rocblas_int k,
                 rocblas_double_complex *A, const rocblas_int lda, double *D, double *E, rocblas_double_complex *tauq, rocblas_double_complex *taup,
                 rocblas_double_complex *X, const rocblas_int ldx, rocblas_double_complex *Y, const rocblas_int ldy)
{
    return rocsolver_labrd_impl<double,rocblas_double_complex>(handle, m, n, k, A, lda, D, E, tauq, taup, X, ldx, Y, ldy);
}

} //extern C
