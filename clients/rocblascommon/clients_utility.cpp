/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/core.h>
#include <rocblas/rocblas.h>

#include "clients_utility.hpp"
#include "rocblas_random.hpp"

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
    int device_count = 0;
    hipError_t status = hipGetDeviceCount(&device_count);
    if(status != hipSuccess)
    {
        fmt::print(stderr, "Query device error: cannot get device count\n");
        return -1;
    }
    fmt::print("Query device success: there are {} devices\n", device_count);

    for(int i = 0;; i++)
    {
        fmt::print("{:-<79}\n", ""); // horizontal rule
        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        status = hipGetDeviceProperties(&props, i);
        if(status != hipSuccess)
        {
            fmt::print(stderr, "Query device error: cannot get device ID {}'s property\n", i);
            continue;
        }

        fmt::print("Device ID {} : {}\nwith {:3.1f} GB memory, max. SCLK {} MHz, "
                   "max. MCLK {} MHz, compute capability {}.{}\nmaxGridDimX {}, "
                   "sharedMemPerBlock {:3.1f} KB, maxThreadsPerBlock {}, warpSize {}\n",
                   i, props.name, props.totalGlobalMem / 1e9, int(props.clockRate / 1000),
                   int(props.memoryClockRate / 1000), props.major, props.minor, props.maxGridSize[0],
                   props.sharedMemPerBlock / 1e3, props.maxThreadsPerBlock, props.warpSize);
    }
    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    hipError_t status = hipSetDevice(device_id);
    if(status != hipSuccess)
        fmt::print(stderr, "Set device error: cannot set device ID {}\n", device_id);
}
