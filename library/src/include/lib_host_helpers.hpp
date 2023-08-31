/* **************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

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

inline size_t idx2D(const size_t i, const size_t j, const size_t inca, const size_t lda)
{
    return j * lda + i * inca;
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
