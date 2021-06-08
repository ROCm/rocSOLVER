/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cstdarg>
#include <ios>
#include <limits>
#include <sstream>

// If USE_ROCBLAS_REALLOC_ON_DEMAND is false, automatic reallocation is disable and we will manually
// reallocate workspace
#define USE_ROCBLAS_REALLOC_ON_DEMAND true

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

typedef enum rocsolver_inform_type_
{
    inform_quick_return,
    inform_invalid_size,
    inform_invalid_args,
    inform_mem_query,
} rocsolver_inform_type;

inline void rocsolver_bench_inform(rocsolver_inform_type it, size_t arg = 0)
{
    switch(it)
    {
    case inform_quick_return: rocsolver_cout << "Quick return..." << std::endl; break;
    case inform_invalid_size: rocsolver_cout << "Invalid size arguments..." << std::endl; break;
    case inform_invalid_args: rocsolver_cout << "Invalid value in arguments..." << std::endl; break;
    case inform_mem_query:
        rocsolver_cout << arg << " bytes of device memory are required..." << std::endl;
        break;
    }
    rocsolver_cout << "No performance data to collect." << std::endl;
    rocsolver_cout << "No computations to verify." << std::endl;
}

inline void rocsolver_bench_output()
{
    // empty version
    rocsolver_cout << std::endl;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    std::stringstream ss;
    ss << std::left << std::setw(15) << arg;

    rocsolver_cout << ss.str();
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
