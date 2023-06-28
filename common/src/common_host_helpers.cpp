/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include <chrono>

#include "common_host_helpers.hpp"

/***********************************************************************
 * timing functions                                                    *
 ***********************************************************************/

/* CPU Timer (in microseconds): no GPU synchronization
 */
double get_time_us_no_sync()
{
    namespace sc = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/* CPU Timer (in microseconds): synchronize with the default device and return wall time
 */
double get_time_us()
{
    hipError_t status = hipDeviceSynchronize();
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}

/* CPU Timer (in microseconds): synchronize with given queue/stream and return wall time
 */
double get_time_us_sync(hipStream_t stream)
{
    hipError_t status = hipStreamSynchronize(stream);
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}
