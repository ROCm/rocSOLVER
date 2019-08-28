#ifndef GETF2_DEVICE_H
#define GETF2_DEVICE_H

#include <hip/hip_runtime.h>

template <typename T, typename U>
__global__ void getf2_check_singularity(U AA, const rocblas_int shiftA, const rocblas_int strideA, 
                                        rocblas_int* ipivA, const rocblas_int shiftP, 
                                        const rocblas_int strideP, const rocblas_int j,
                                        const rocblas_int lda,
                                        T* invpivot, rocblas_int* info)
{
    int id = hipBlockIdx_x;

    T* A;
    #ifdef batched
        A = AA[id] + shiftA;
    #else
        A = AA + id*strideA + shiftA;     
    #endif 
    rocblas_int *ipiv = ipivA + id*strideP + shiftP;

    ipiv[j] += j;           //update the pivot index
    if (A[j * lda + ipiv[j] - 1] == 0) {
        invpivot[id] = 1.0;
        if (info[id] == 0)
           info[id] = j + 1;   //use Fortran 1-based indexing
    }
    else 
        invpivot[id] = 1.0 / A[j * lda + ipiv[j] - 1];
}


__global__ void getrf_check_singularity(const rocblas_int n, const rocblas_int j, rocblas_int *ipivA, const rocblas_int shiftP, 
                                const rocblas_int strideP, const rocblas_int *iinfo, rocblas_int *info) {
    int id = hipBlockIdx_y;
    
    rocblas_int *ipiv = ipivA + id*strideP + shiftP;

    if (info[id] == 0 && iinfo[id] > 0)
        info[id] = iinfo[id] + j;
        
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (tid < n) 
        ipiv[tid] += j;
}

    


#endif /*GETF2_DEVICE_H */
