/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */


#ifndef S_TEST_H_
#define S_TEST_H_

#include <limits>
#include <boost/format.hpp>
#include <cstdarg>

#define ROCSOLVER_BENCH_INFORM(case)                                        \
    do                                                                      \
    {                                                                       \
        if (case == 2)                                                      \
            rocblas_cout << "Invalid value in arguments ..." << std::endl;  \
        else if (case == 1)                                                 \
            rocblas_cout << "Invalid size arguments..." << std::endl;       \
        else                                                                \
            rocblas_cout << "Quick return..." << std::endl;                 \
        rocblas_cout << "No performance data to collect." << std::endl;     \
        rocblas_cout << "No computations to verify." << std::endl;          \
    } while(0)



template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
constexpr double get_epsilon()
{
    return std::numeric_limits<T>::epsilon();
}

template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
constexpr auto get_epsilon()
{
    return get_epsilon<decltype(std::real(T{}))>();
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
    rocblas_cout << std::endl << std::flush;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    using boost::format;
    format f("%|-10|");    

    rocblas_cout << f % arg;
    rocsolver_bench_output(args...);
}

#endif
