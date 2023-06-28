/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution
 * ************************************************************************/

#pragma once

#include <cassert>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "lib_macros.hpp"
#include "rocsolver_logger.hpp"

#define IOTA_MAX_THDS 32

// Fills the given range with sequentially increasing values.
// The name and interface is based on std::iota
template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(IOTA_MAX_THDS) iota_n(T* first, uint32_t count, T value)
{
    const auto idx = hipThreadIdx_x;
    if(idx < count)
    {
        first[idx] = T(idx) + value;
    }
}

// Initializes scalars on the device.
template <typename T>
void init_scalars(rocblas_handle handle, T* scalars)
{
    assert(scalars != nullptr);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    ROCSOLVER_LAUNCH_KERNEL(iota_n<T>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream, scalars, 3, -1);
}
