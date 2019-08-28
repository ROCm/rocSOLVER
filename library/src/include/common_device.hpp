#ifndef COMMON_DEVICE_H
#define COMMON_DEVICE_H

#include <hip/hip_runtime.h>


__global__ void reset_info(rocblas_int *info, const rocblas_int n) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < n)
        info[idx] = 0;
}

#endif
