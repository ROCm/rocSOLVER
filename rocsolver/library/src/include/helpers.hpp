/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HELPERS_H
#define HELPERS_H

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>

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

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
constexpr double get_safemin()
{
    auto eps = get_epsilon<T>();
    auto s1 = std::numeric_limits<T>::min();
    auto s2 = 1 / std::numeric_limits<T>::max();
    if(s2 > s1)
        return s2 * (1 + eps);
    return s1;
}

template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
constexpr auto get_safemin()
{
    return get_safemin<decltype(std::real(T{}))>();
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
    T const* R = array;
    return R;
}

template <typename T>
T const* const* cast2constType(T* const* array)
{
    T const* const* R = array;
    return R;
}

template <typename T>
T* cast2constPointer(T* array)
{
    T* R = array;
    return R;
}

template <typename T>
T* const* cast2constPointer(T** array)
{
    T* const* R = array;
    return R;
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

// ROCSOLVER_UNREACHABLE is an alternative to __builtin_unreachable
// that verifies that the path is actually unreachable if
// ROCSOLVER_VERIFY_ASSUMPTIONS is defined.
#ifdef ROCSOLVER_VERIFY_ASSUMPTIONS
#define ROCSOLVER_UNREACHABLE std::abort
#else
#define ROCSOLVER_UNREACHABLE __builtin_unreachable
#endif

#endif /* HELPERS_H */
