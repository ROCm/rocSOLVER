/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdio>
#include <string>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <rocsolver/rocsolver.h>

#include "clientcommon.hpp"

static std::string rocblas_version()
{
    size_t size;
    rocblas_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocblas_get_version_string(str.data(), size);
    return str;
}

static std::string rocsolver_version()
{
    size_t size;
    rocsolver_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocsolver_get_version_string(str.data(), size);
    return str;
}

static void print_version_info()
{
    fmt::print("rocSOLVER version {} (with rocBLAS {})\n", rocsolver_version(), rocblas_version());
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
