/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdio>
#include <fmt/core.h>
#include <gtest/gtest.h>

#include "clientcommon.hpp"
#include "rocblas/internal/rocblas-version.h"
#include "rocsolver/rocsolver-version.h"

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

static void print_version_info()
{
    fmt::print("rocSOLVER version {}.{}.{}.{} (with rocBLAS {}.{}.{}.{})\n",
               STRINGIFY(ROCSOLVER_VERSION_MAJOR), STRINGIFY(ROCSOLVER_VERSION_MINOR),
               STRINGIFY(ROCSOLVER_VERSION_PATCH), STRINGIFY(ROCSOLVER_VERSION_TWEAK),
               STRINGIFY(ROCBLAS_VERSION_MAJOR), STRINGIFY(ROCBLAS_VERSION_MINOR),
               STRINGIFY(ROCBLAS_VERSION_PATCH), STRINGIFY(ROCBLAS_VERSION_TWEAK));
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    print_version_info();

    // print device info
    int device_count = query_device_property();
    if(device_count <= 0)
    {
        fmt::print(stderr, "Error: No devices found\n");
        return -1;
    }
    set_device(0); // use first device

    // Initialize gtest and rocBLAS
    ::testing::InitGoogleTest(&argc, argv);
    rocblas_initialize();

    int status = RUN_ALL_TESTS();
    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
