/* ************************************************************************
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/core.h>
#include <rocblas/rocblas.h>

#include "clients_utility.hpp"
#include "rocblas_random.hpp"

#ifdef _WIN32

#include <libloaderapi.h>

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#else
#include <fcntl.h>
#endif

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

// Return the path to the currently running executable
std::string rocsolver_exepath()
{
#ifdef _WIN32

    std::vector<TCHAR> result(MAX_PATH + 1);
    DWORD length = 0;
    for(;;)
    {
        length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            break;
        }
        result.resize(result.size() * 2);
    }

    fs::path exepath(result.begin(), result.end());
    exepath = exepath.remove_filename();
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();

#else
    std::string pathstr;
    char* path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1] = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

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
