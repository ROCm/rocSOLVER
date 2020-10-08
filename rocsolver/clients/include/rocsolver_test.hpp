/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef S_TEST_H_
#define S_TEST_H_

#include <boost/format.hpp>
#include <cstdarg>
#include <limits>

#define ROCSOLVER_BENCH_INFORM(case)                                       \
    do                                                                     \
    {                                                                      \
        if(case == 2)                                                      \
            rocblas_cout << "Invalid value in arguments ..." << std::endl; \
        else if(case == 1)                                                 \
            rocblas_cout << "Invalid size arguments..." << std::endl;      \
        else                                                               \
            rocblas_cout << "Quick return..." << std::endl;                \
        rocblas_cout << "No performance data to collect." << std::endl;    \
        rocblas_cout << "No computations to verify." << std::endl;         \
    } while(0)

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

template <typename T>
inline void rocsolver_test_check(double max_error, int tol)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, tol * get_epsilon<T>());
#endif
}

inline void rocsolver_bench_output()
{
    // empty version
    rocblas_cout << std::endl;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    using boost::format;
    format f("%|-15|");

    rocblas_cout << f % arg;
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

#endif
