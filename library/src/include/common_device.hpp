#ifndef COMMON_DEVICE_H
#define COMMON_DEVICE_H

#include <hip/hip_runtime.h>

template<typename T, typename U>
__forceinline__ __global__ void reset_info(T *info, const rocblas_int n, U val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (idx < n)
        info[idx] = T(val);
}

template<typename T, typename U>
__forceinline__ __global__ void reset_batch_info(T *info, const rocblas_int stride, const rocblas_int n, U val) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int b = hipBlockIdx_y;

    T* inf = info + b * stride;
    if (idx < n)
        inf[idx] = T(val);
}

template<typename T>
__forceinline__ __device__ __host__ T* load_ptr_batch(T* p, rocblas_int shift, rocblas_int batch, rocblas_int stride) {
    return p + batch * stride + shift;
}

template<typename T>
__forceinline__ __device__ __host__ T* load_ptr_batch(T *const p[], rocblas_int shift, rocblas_int batch, rocblas_int stride) {
    return p[batch] + shift;
}

template<typename T>
__forceinline__ __global__ void get_array(T** out, T* in, rocblas_int stride, rocblas_int batch) 
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    if (b < batch)
        out[b] = in + b*stride;
}


#endif
