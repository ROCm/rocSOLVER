/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <limits>

// If USE_ROCBLAS_REALLOC_ON_DEMAND is false, automatic reallocation is disable and we will manually
// reallocate workspace
#define USE_ROCBLAS_REALLOC_ON_DEMAND true

#define ROCSOLVER_BENCH_INFORM(case)                                         \
    do                                                                       \
    {                                                                        \
        if(case == 2)                                                        \
            rocsolver_cout << "Invalid value in arguments ..." << std::endl; \
        else if(case == 1)                                                   \
            rocsolver_cout << "Invalid size arguments..." << std::endl;      \
        else                                                                 \
            rocsolver_cout << "Quick return..." << std::endl;                \
        rocsolver_cout << "No performance data to collect." << std::endl;    \
        rocsolver_cout << "No computations to verify." << std::endl;         \
    } while(0)

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

#ifdef GOOGLE_TEST
#define ROCSOLVER_TEST_CHECK(T, max_error, tol) ASSERT_LE((max_error), (tol)*get_epsilon<T>())
#else
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)
#endif

// format strings for rocsolver_bench_output
template <typename T>
static constexpr auto rocsolver_bench_specifier = "";
template <>
static constexpr auto rocsolver_bench_specifier<int> = "i";
template <>
static constexpr auto rocsolver_bench_specifier<long> = "li";
template <>
static constexpr auto rocsolver_bench_specifier<double> = "g";
template <>
static constexpr auto rocsolver_bench_specifier<char> = "c";
template <>
static constexpr auto rocsolver_bench_specifier<const char*> = "s";

inline void rocsolver_bench_output()
{
    // empty version
    rocsolver_cout << std::endl;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    // create format string for given BUF_SIZE and type T
    static const int BUF_SIZE = 15;
    std::string format("%-");
    format += std::to_string(BUF_SIZE);
    format += rocsolver_bench_specifier<T>;

    // create string buffer
    static char buffer[BUF_SIZE + 1];

    // format string (with trailing ellipsis if needed)
    int total = snprintf(buffer, BUF_SIZE + 1, format.c_str(), arg);
    if(total > BUF_SIZE)
        buffer[BUF_SIZE - 1] = buffer[BUF_SIZE - 2] = buffer[BUF_SIZE - 3] = '.';

    // print
    rocsolver_cout << buffer;
    if(sizeof...(Ts) > 0)
        rocsolver_cout << ' ';
    rocsolver_bench_output(args...);
}

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return scalar;
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return std::conj(scalar);
}

// A struct implicity convertable to and from char, used so we can customize
// Google Test printing for LAPACK char arguments without affecting the default
// char output.
struct rocsolver_op_char
{
    rocsolver_op_char(char c)
        : data(c)
    {
    }

    operator char() const
    {
        return data;
    }

    char data;
};

// gtest printers

inline std::ostream& operator<<(std::ostream& os, rocblas_status x)
{
    return os << rocblas_status_to_string(x);
}

inline std::ostream& operator<<(std::ostream& os, rocsolver_op_char x)
{
    return os << x.data;
}
