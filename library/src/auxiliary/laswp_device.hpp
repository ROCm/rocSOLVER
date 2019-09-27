#ifndef LASWP_DEVICE_H
#define LASWP_DEVICE_H

#include <hip/hip_runtime.h>


template <typename T>
__device__ void swap(const rocblas_int n, T *a, const rocblas_int lda,
                               const rocblas_int i,
                               const rocblas_int exch) {

    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid < n) {
        T orig = a[i + lda * tid];
        a[i + lda * tid] = a[exch + lda * tid];
        a[exch + lda * tid] = orig;
    }
}

template <typename T, typename U>
__global__ void laswp_kernel(const rocblas_int n, U AA, const rocblas_int shiftA,
                            const rocblas_int lda, const rocblas_int stride, const rocblas_int i, const rocblas_int k1,
                            const rocblas_int *ipivA, const rocblas_int shiftP, const rocblas_int strideP, const rocblas_int incx) {

    int id = hipBlockIdx_y;

    //shiftP must be used so that ipiv[k1] is the desired first index of ipiv    
    const rocblas_int *ipiv = ipivA + id*strideP + shiftP;
    rocblas_int exch = ipiv[k1 + (i - k1) * incx - 1];

    //will exchange rows i and exch if they are not the same
    if (exch != i) {
        T* A;
        #ifdef batched
            A = AA[id] + shiftA;
        #else
            A = AA + id*stride + shiftA;
        #endif

        swap(n,A,lda,i-1,exch-1);  //row indices are base-1 from the API
    }
}

#endif
