/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "clientcommon.hpp"
#include "internal/rocblas-version.h"
#include "rocsolver-version.h"
#include <gtest/gtest.h>
#include <stdexcept>

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

static void print_version_info()
{
    // clang-format off
    rocblas_cout << "rocSOLVER version "
        STRINGIFY(ROCSOLVER_VERSION_MAJOR) "."
        STRINGIFY(ROCSOLVER_VERSION_MINOR) "."
        STRINGIFY(ROCSOLVER_VERSION_PATCH) "."
        STRINGIFY(ROCSOLVER_VERSION_TWEAK)
        " (with rocBLAS "
        STRINGIFY(ROCBLAS_VERSION_MAJOR) "."
        STRINGIFY(ROCBLAS_VERSION_MINOR) "."
        STRINGIFY(ROCBLAS_VERSION_PATCH) "."
        STRINGIFY(ROCBLAS_VERSION_TWEAK) ")"
        << std::endl;
    // clang-format on
}

int main(int argc, char** argv)
{
    print_version_info();

    // Device Query
    int device_id = 0;
    int device_count = query_device_property();
    if(device_count <= device_id)
    {
        rocblas_cerr << "Error: invalid device ID. There may not be such device ID." << std::endl;
        return -1;
    }
    set_device(device_id);

    // Initialize gtest and rocBLAS
    ::testing::InitGoogleTest(&argc, argv);
    rocblas_initialize();

    int status = RUN_ALL_TESTS();
    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
