/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

// Suppress warnings about hipMalloc(), hipFree() except in rocblas-test and
// rocblas-bench
#if !defined(ROCSOLVER_CLIENTS_TEST) && !defined(ROCBLAS_BENCH)
#undef hipMalloc
#undef hipFree
#endif

#ifdef ROCSOLVER_CLIENTS_TEST
#include <gtest/gtest.h>

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                                                     \
    do                                                                                     \
    {                                                                                      \
        auto error = ERROR;                                                                \
        if(error == hipErrorOutOfMemory)                                                   \
        {                                                                                  \
            SUCCEED() << LIMITED_MEMORY_STRING;                                            \
            return;                                                                        \
        }                                                                                  \
        else if(error != hipSuccess)                                                       \
        {                                                                                  \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, \
                    __FILE__, __LINE__);                                                   \
            return;                                                                        \
        }                                                                                  \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                  \
    do                                                             \
    {                                                              \
        auto status__ = (STATUS);                                  \
        ASSERT_TRUE(status__ == rocblas_status_size_increased      \
                    || status__ == rocblas_status_size_unchanged); \
    } while(0)

#define EXPECT_ROCBLAS_STATUS ASSERT_EQ

#else // ROCSOLVER_CLIENTS_TEST

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        fmt::print(stderr, "rocBLAS status error: Expected {}, received {}\n",
                   rocblas_status_to_string(expect), rocblas_status_to_string(status));
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP_ERROR(ERROR)                                                               \
    do                                                                                       \
    {                                                                                        \
        auto error = ERROR;                                                                  \
        if(error != hipSuccess)                                                              \
        {                                                                                    \
            fmt::print(stderr, "error: {} ({}) at {}:{}\n", hipGetErrorString(error), error, \
                       __FILE__, __LINE__);                                                  \
            rocblas_abort();                                                                 \
        }                                                                                    \
    } while(0)

#define CHECK_ALLOC_QUERY(STATUS)                                                                     \
    do                                                                                                \
    {                                                                                                 \
        auto status__ = (STATUS);                                                                     \
        if(!(status__ == rocblas_status_size_increased || status__ == rocblas_status_size_unchanged)) \
        {                                                                                             \
            fmt::print(stderr,                                                                        \
                       "rocBLAS status error: Expected rocblas_status_size_unchanged or "             \
                       "rocblas_status_size_increase,\nreceived {}\n",                                \
                       rocblas_status_to_string(status__));                                           \
            rocblas_abort();                                                                          \
        }                                                                                             \
    } while(0)

#define CHECK_DEVICE_ALLOCATION CHECK_HIP_ERROR

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#endif // ROCSOLVER_CLIENTS_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)
