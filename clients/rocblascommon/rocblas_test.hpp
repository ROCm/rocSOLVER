/* ************************************************************************
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

struct rocsolver_info_accumulator
{
    template <typename T>
    rocsolver_info_accumulator& operator<<(T&&)
    {
        // todo: implement this so rocsolver-bench can print extra
        //       info about failures when doing error checking.
        return *this;
    }
};

struct rocsolver_expect_eq : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_eq(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 == v2))
        {
            fmt::print(stderr, "{}:{}: expected {} == {}!\n", file, line, v1, v2);
        }
    }
};

struct rocsolver_expect_ne : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_ne(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 != v2))
        {
            fmt::print(stderr, "{}:{}: expected {} != {}!\n", file, line, v1, v2);
        }
    }
};

struct rocsolver_expect_lt : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_lt(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 < v2))
        {
            fmt::print(stderr, "{}:{}: expected {} < {}!\n", file, line, v1, v2);
        }
    }
};

struct rocsolver_expect_le : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_le(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 <= v2))
        {
            fmt::print(stderr, "{}:{}: expected {} <= {}!\n", file, line, v1, v2);
        }
    }
};

struct rocsolver_expect_gt : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_gt(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 > v2))
        {
            fmt::print(stderr, "{}:{}: expected {} > {}!\n", file, line, v1, v2);
        }
    }
};

struct rocsolver_expect_ge : rocsolver_info_accumulator
{
    template <typename T1, typename T2>
    rocsolver_expect_ge(T1&& v1, T2&& v2, fmt::string_view file, int line)
    {
        if(!(v1 >= v2))
        {
            fmt::print(stderr, "{}:{}: expected {} >= {}!\n", file, line, v1, v2);
        }
    }
};

#define EXPECT_EQ(v1, v2) rocsolver_expect_eq(v1, v2, __FILE__, __LINE__)
#define EXPECT_NE(v1, v2) rocsolver_expect_ne(v1, v2, __FILE__, __LINE__)
#define EXPECT_LT(v1, v2) rocsolver_expect_lt(v1, v2, __FILE__, __LINE__)
#define EXPECT_LE(v1, v2) rocsolver_expect_le(v1, v2, __FILE__, __LINE__)
#define EXPECT_GT(v1, v2) rocsolver_expect_gt(v1, v2, __FILE__, __LINE__)
#define EXPECT_GE(v1, v2) rocsolver_expect_ge(v1, v2, __FILE__, __LINE__)

#endif // ROCSOLVER_CLIENTS_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)
