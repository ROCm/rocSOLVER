/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

/* The format function for user-defined types cannot be const before fmt v8.0
   but must be const in fmt v8.1 if the type is used in a tuple. */
#if FMT_VERSION < 80000
#define ROCSOLVER_FMT_CONST
#else
#define ROCSOLVER_FMT_CONST const
#endif

namespace fmt
{
template <typename T>
struct formatter<rocblas_complex_num<T>> : formatter<T>
{
    template <typename FormatCtx>
    auto format(const rocblas_complex_num<T>& value, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        formatter<T>::format(value.real(), ctx);
        format_to(ctx.out(), "+");
        formatter<T>::format(value.imag(), ctx);
        format_to(ctx.out(), "*i");
        return ctx.out();
    }
};
}
