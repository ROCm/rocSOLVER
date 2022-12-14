/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

/*
 * ===========================================================================
 *    common location for functions that are used across several rocSOLVER
 *    routines, excepting device functions and kernels (see
 *    lib_device_helpers.hpp and lapack_device_functions.hpp).
 * ===========================================================================
 */

inline size_t idx2D(const size_t i, const size_t j, const size_t lda)
{
    return j * lda + i;
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

inline rocblas_int get_index(rocblas_int* intervals, rocblas_int max, rocblas_int dim)
{
    rocblas_int i;

    for(i = 0; i < max; ++i)
    {
        if(dim <= intervals[i])
            break;
    }

    return i;
}

/** FIND_MAX_TRIDIAG finds the element with the largest magnitude in the
    tridiagonal matrix **/
template <typename T>
T host_find_max_tridiag(const rocblas_int start, const rocblas_int end, T* D, T* E)
{
    T anorm = abs(D[end]);
    for(int i = start; i < end; i++)
        anorm = max(anorm, max(abs(D[i]), abs(E[i])));
    return anorm;
}

/** SCALE_TRIDIAG scales the elements of the tridiagonal matrix by a given
    scale factor **/
template <typename T>
void host_scale_tridiag(const rocblas_int start, const rocblas_int end, T* D, T* E, T scale)
{
    D[end] *= scale;
    for(int i = start; i < end; i++)
    {
        D[i] *= scale;
        E[i] *= scale;
    }
}

/** LAE2 computes the eigenvalues of a 2x2 symmetric matrix
    [ a b ]
    [ b c ] **/
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void host_lae2(T& a, T& b, T& c, T& rt1, T& rt2)
{
    T sm = a + c;
    T adf = abs(a - c);
    T ab = abs(b + b);

    T rt, acmx, acmn;
    if(adf > ab)
    {
        rt = ab / adf;
        rt = adf * sqrt(1 + rt * rt);
    }
    else if(adf < ab)
    {
        rt = adf / ab;
        rt = ab * sqrt(1 + rt * rt);
    }
    else
        rt = ab * sqrt(2);

    // Compute the eigenvalues
    if(abs(a) > abs(c))
    {
        acmx = a;
        acmn = c;
    }
    else
    {
        acmx = c;
        acmn = a;
    }
    if(sm < 0)
    {
        rt1 = T(0.5) * (sm - rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else if(sm > 0)
    {
        rt1 = T(0.5) * (sm + rt);
        rt2 = T((acmx / (double)rt1) * acmn - (b / (double)rt1) * b);
    }
    else
    {
        rt1 = T(0.5) * rt;
        rt2 = T(-0.5) * rt;
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
