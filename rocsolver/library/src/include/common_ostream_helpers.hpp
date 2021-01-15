/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "libcommon.hpp"
#include "rocsolver.h"

/*
 * ===========================================================================
 *    common location for functions that are used to output rocsolver data
 *    types (e.g. for logging purposes).
 * ===========================================================================
 */

template <typename T, typename... Ts>
void print_list(rocsolver_ostream& os, const char* sep, T arg, Ts... args)
{
    os << arg;

    if(sizeof...(Ts) > 0)
    {
        os << sep;
        print_list(os, sep, args...);
    }
}
inline void print_list(rocsolver_ostream& os, const char* sep)
{
    // do nothing
}

template <typename T1, typename T2, typename... Ts>
void print_pairs(rocsolver_ostream& os, const char* sep, T1 arg1, T2 arg2, Ts... args)
{
    os << arg1 << ' ' << arg2;

    if(sizeof...(Ts) > 0)
    {
        os << sep;
        print_pairs(os, sep, args...);
    }
}
inline void print_pairs(rocsolver_ostream& os, const char* sep)
{
    // do nothing
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
