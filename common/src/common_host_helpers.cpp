/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "common_host_helpers.hpp"

/***********************************************************************
 * timing functions                                                    *
 ***********************************************************************/

namespace ROCSOLVER_COMMON_NAMESPACE {

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and
 * return wall time */
double get_time_us()
{
    hipDeviceSynchronize();
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and
 * return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

}
