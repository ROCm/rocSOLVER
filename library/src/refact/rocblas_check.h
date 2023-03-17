
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef ROCBLAS_CHECK_H
#define ROCBLAS_CHECK_H

#include <exception>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include "rocblas/rocblas.h"

#ifndef rocblasGetErrorName
#define rocblasGetErrorName(istat)                                                                \
    (((istat) == rocblas_status_success)                   ? "rocblas_status_success"             \
         : ((istat) == rocblas_status_invalid_handle)      ? "rocblas_status_invalid_handle"      \
         : ((istat) == rocblas_status_not_implemented)     ? "rocblas_status_not_implemented"     \
         : ((istat) == rocblas_status_invalid_pointer)     ? "rocblas_status_invalid_pointer"     \
         : ((istat) == rocblas_status_invalid_size)        ? "rocblas_status_invalid_size"        \
         : ((istat) == rocblas_status_memory_error)        ? "rocblas_status_memory_error"        \
         : ((istat) == rocblas_status_internal_error)      ? "rocblas_status_internal_error"      \
         : ((istat) == rocblas_status_perf_degraded)       ? "rocblas_status_perf_degraded"       \
         : ((istat) == rocblas_status_size_query_mismatch) ? "rocblas_status_size_query_mismatch" \
         : ((istat) == rocblas_status_size_increased)      ? "rocblas_status_size_increased"      \
         : ((istat) == rocblas_status_size_unchanged)      ? "rocblas_status_size_unchanged"      \
         : ((istat) == rocblas_status_invalid_value)       ? "rocblas_status_invalid_value"       \
         : ((istat) == rocblas_status_continue)            ? "rocblas_status_continue"            \
         : ((istat) == rocblas_status_check_numerics_fail) ? "rocblas_status_check_numerics_fail" \
                                                           : "unknown status code")
#endif

#ifndef ROCBLAS_CHECK
#define ROCBLAS_CHECK(fcn, error_code)                                                        \
    {                                                                                         \
        rocblas_status const istat = (fcn);                                                   \
        if(istat != rocblas_status_success)                                                   \
        {                                                                                     \
            printf("rocblas API failed at line %d in file %s with error: %s(%d)\n", __LINE__, \
                   __FILE__, rocblasGetErrorName(istat), istat);                              \
            fflush(stdout);                                                                   \
            return ((error_code));                                                            \
        };                                                                                    \
    };
#endif

#ifndef THROW_IF_ROCBLAS_ERROR
#define THROW_IF_ROCBLAS_ERROR(fcn)                                                    \
    {                                                                                  \
        rocblas_status const istat = (fcn);                                            \
        if(istat != rocblas_status_success)                                            \
        {                                                                              \
            printf("rocblas failed at %s:%d, with error %s(%d)\n", __FILE__, __LINE__, \
                   rocblasGetErrorName(istat), istat);                                 \
            fflush(stdout);                                                            \
            throw std::runtime_error(__FILE__);                                        \
        };                                                                             \
    };

#endif

#endif
