/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "common_ostream_helpers.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
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

    rocblas_cout << m << "-by-" << n << " matrix: " << name << std::endl;
    for(int i = 0; i < m; i++)
    {
        rocblas_cout << "    ";
        for(int j = 0; j < n; j++)
        {
            rocblas_cout << hA[j * lda + i];
            if(j < n - 1)
                rocblas_cout << ", ";
        }
        rocblas_cout << std::endl;
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

    rocblas_cout << m << "-by-" << n << " matrix: " << name << std::endl;
    for(int i = 0; i < m; i++)
    {
        rocblas_cout << "    ";
        for(int j = 0; j < n; j++)
        {
            rocblas_cout << '[' << hA[j * lda + i].real() << "+" << hA[j * lda + i].imag() << "i]";
            if(j < n - 1)
                rocblas_cout << ", ";
        }
        rocblas_cout << std::endl;
    }
}

#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
// Ensure __assert_fail is declared.
#if !__is_identifier(__assert_fail)
extern "C" [[noreturn]] void __assert_fail(const char* assertion,
                                           const char* file,
                                           unsigned int line,
                                           const char* function) noexcept;
#endif
// ROCSOLVER_FAIL(msg) is called with a string literal to print a message and abort the program.
// By default, it calls __assert_fail, but can be defined to something else.
#ifndef ROCSOLVER_FAIL
#define ROCSOLVER_FAIL(msg) __assert_fail(msg, __FILE__, __LINE__, __PRETTY_FUNCTION__)
#endif
#endif

// ROCSOLVER_UNREACHABLE is an alternative to __builtin_unreachable that verifies that the path is
// actually unreachable if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE() ROCSOLVER_FAIL("unreachable statement")
#else
#define ROCSOLVER_UNREACHABLE() __builtin_unreachable()
#endif

// ROCSOLVER_UNREACHABLE_X is a variant of ROCSOLVER_UNREACHABLE that takes a string as a parameter,
// which should explain why this path is believed to be unreachable.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE_X(msg) ROCSOLVER_FAIL("unreachable statement (assumed " msg ")")
#else
#define ROCSOLVER_UNREACHABLE_X(msg) __builtin_unreachable()
#endif

// ROCSOLVER_ASSUME is an alternative to __builtin_assume that verifies that the assumption is
// actually true if ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_ASSUME(invariant)     \
    do                                  \
    {                                   \
        if(!(invariant))                \
        {                               \
            ROCSOLVER_FAIL(#invariant); \
        }                               \
    } while(0)
#else
#define ROCSOLVER_ASSUME(invariant) __builtin_assume(invariant)
#endif

// ROCSOLVER_ASSUME_X is a variant of ROCSOLVER_ASSUME that takes a string as a second parameter,
// which should explain why this invariant is believed to be guaranteed.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_ASSUME_X(invariant, msg)                   \
    do                                                       \
    {                                                        \
        if(!(invariant))                                     \
        {                                                    \
            ROCSOLVER_FAIL(#invariant " (assumed " msg ")"); \
        }                                                    \
    } while(0)
#else
#define ROCSOLVER_ASSUME_X(invariant, msg) __builtin_assume(invariant)
#endif
