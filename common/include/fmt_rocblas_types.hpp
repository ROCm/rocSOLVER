/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
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
