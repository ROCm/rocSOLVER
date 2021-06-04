/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "utility.hpp"
#include "rocblas_random.hpp"
#include <cstdlib>
#include <cstring>

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
const rocblas_rng_t rocblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
const std::thread::id main_thread_id = std::this_thread::get_id();

// For the main thread, we use rocblas_seed; for other threads, we start with a
// different seed but deterministically based on the thread id's hash function.
thread_local rocblas_rng_t rocblas_rng = get_seed();

/* ============================================================================================
 */
/*  device query and print out their ID and name; return number of
 * compute-capable devices. */
rocblas_int query_device_property()
{
    int device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        rocsolver_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        rocsolver_cout << "Query device success: there are " << device_count << " devices"
                       << std::endl;
    }

    for(rocblas_int i = 0;; i++)
    {
        rocsolver_cout << "----------------------------------------------------------"
                          "---------------------"
                       << std::endl;

        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        rocblas_status status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            rocsolver_cerr << "Query device error: cannot get device ID " << i << "'s property"
                           << std::endl;
        }
        else
        {
            char buf[320];
            snprintf(buf, sizeof(buf),
                     "Device ID %d : %s\n"
                     "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, "
                     "compute capability "
                     "%d.%d\n"
                     "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock "
                     "%d, warpSize %d\n",
                     i, props.name, props.totalGlobalMem / 1e9, (int)(props.clockRate / 1000),
                     (int)(props.memoryClockRate / 1000), props.major, props.minor,
                     props.maxGridSize[0], props.sharedMemPerBlock / 1e3, props.maxThreadsPerBlock,
                     props.warpSize);
            rocsolver_cout << buf;
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        rocsolver_cerr << "Set device error: cannot set device ID " << device_id
                       << ", there may not be such device ID" << std::endl;
    }
}
