/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef COMMON_HOST_HELPERS_H
#define COMMON_HOST_HELPERS_H

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <rocblas.h>

/*
 * ===========================================================================
 *    common location for functions that are used across several rocSOLVER
 *    routines, excepting device functions and kernels (see
 *    common_device_helpers.hpp and lapack_device_functions.hpp).
 * ===========================================================================
 */

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

template <typename T>
constexpr double get_safemin()
{
    using S = decltype(std::real(T{}));
    auto eps = get_epsilon<S>();
    auto s1 = std::numeric_limits<S>::min();
    auto s2 = 1 / std::numeric_limits<S>::max();
    if(s2 > s1)
        return s2 * (1 + eps);
    return s1;
}

inline size_t idx2D(const size_t i, const size_t j, const size_t lda)
{
    return j * lda + i;
}

template <typename T>
inline T machine_precision();
template <>
inline float machine_precision()
{
    return static_cast<float>(1.19e-07);
}
template <>
inline double machine_precision()
{
    return static_cast<double>(2.22e-16);
}

template <typename T>
T const* cast2constType(T* array)
{
    return array;
}

template <typename T>
T const* const* cast2constType(T* const* array)
{
    return array;
}

template <typename T>
T* cast2constPointer(T* array)
{
    return array;
}

template <typename T>
T* const* cast2constPointer(T** array)
{
    return array;
}

template <typename T, typename U, std::enable_if_t<!is_complex<T>, int> = 0>
void print_device_matrix(const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         U A,
                         const rocblas_int lda)
{
    T hA[lda * n];
    hipMemcpy(hA, A, sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    std::cerr << m << "-by-" << n << " matrix: " << name << '\n';
    for(int i = 0; i < m; i++)
    {
        std::cerr << "    ";
        for(int j = 0; j < n; j++)
        {
            std::cerr << hA[j * lda + i];
            if(j < n - 1)
                std::cerr << ", ";
        }
        std::cerr << '\n';
    }
}

template <typename T, typename U, std::enable_if_t<is_complex<T>, int> = 0>
void print_device_matrix(const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         U A,
                         const rocblas_int lda)
{
    T hA[lda * n];
    hipMemcpy(hA, A, sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    std::cerr << m << "-by-" << n << " matrix: " << name << '\n';
    for(int i = 0; i < m; i++)
    {
        std::cerr << "    ";
        for(int j = 0; j < n; j++)
        {
            std::cerr << '[' << hA[j * lda + i].real() << "+" << hA[j * lda + i].imag() << "i]";
            if(j < n - 1)
                std::cerr << ", ";
        }
        std::cerr << '\n';
    }
}

// ROCSOLVER_UNREACHABLE is an alternative to __builtin_unreachable that verifies that the path is
// actually unreachable if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE std::abort
#else
#define ROCSOLVER_UNREACHABLE __builtin_unreachable
#endif

// ROCSOLVER_UNREACHABLE_X is a variant of ROCSOLVER_UNREACHABLE that takes a string as a parameter,
// which should explain why this path is believed to be unreachable.
#define ROCSOLVER_UNREACHABLE_X(message) ROCSOLVER_UNREACHABLE()

// ROCSOLVER_ASSUME is an alternative to __builtin_assume that verifies that the assumption is
// actually true if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_ASSUME(invariant) \
    do                              \
    {                               \
        if(!(invariant))            \
            std::abort();           \
    } while(0)
#else
#define ROCSOLVER_ASSUME(invariant) __builtin_assume(invariant)
#endif

// ROCSOLVER_ASSUME_X is a variant of ROCSOLVER_ASSUME that takes a string as a second parameter,
// which should explain why this invariant is believed to be guaranteed.
#define ROCSOLVER_ASSUME_X(invariant, message) ROCSOLVER_ASSUME(invariant)

#endif /* COMMON_HOST_HELPERS_H */
