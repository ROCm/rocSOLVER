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

constexpr char rocblas_direct_letter(rocblas_direct value)
{
    switch(value)
    {
    case rocblas_forward_direction: return 'F';
    case rocblas_backward_direction: return 'B';
    }
    return ' ';
}

constexpr char rocblas_storev_letter(rocblas_storev value)
{
    switch(value)
    {
    case rocblas_column_wise: return 'C';
    case rocblas_row_wise: return 'R';
    }
    return ' ';
}

constexpr char rocblas_workmode_letter(rocblas_workmode value)
{
    switch(value)
    {
    case rocblas_outofplace: return 'O';
    case rocblas_inplace: return 'I';
    }
    return ' ';
}

constexpr char rocblas_svect_letter(rocblas_svect value)
{
    switch(value)
    {
    case rocblas_svect_all: return 'A';
    case rocblas_svect_singular: return 'S';
    case rocblas_svect_overwrite: return 'O';
    case rocblas_svect_none: return 'N';
    }
    return ' ';
}

constexpr char rocblas_evect_letter(rocblas_evect value)
{
    switch(value)
    {
    case rocblas_evect_original: return 'V';
    case rocblas_evect_tridiagonal: return 'I';
    case rocblas_evect_none: return 'N';
    }
    return ' ';
}

inline rocsolver_ostream& operator<<(rocsolver_ostream& os, rocblas_direct value)
{
    return rocsolver_ostream::cout() << rocblas_direct_letter(value);
}

inline rocsolver_ostream& operator<<(rocsolver_ostream& os, rocblas_storev value)
{
    return rocsolver_ostream::cout() << rocblas_storev_letter(value);
}

inline rocsolver_ostream& operator<<(rocsolver_ostream& os, rocblas_workmode value)
{
    return rocsolver_ostream::cout() << rocblas_workmode_letter(value);
}

inline rocsolver_ostream& operator<<(rocsolver_ostream& os, rocblas_svect value)
{
    return rocsolver_ostream::cout() << rocblas_svect_letter(value);
}

inline rocsolver_ostream& operator<<(rocsolver_ostream& os, rocblas_evect value)
{
    return rocsolver_ostream::cout() << rocblas_evect_letter(value);
}
