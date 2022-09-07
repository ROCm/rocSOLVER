/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "fmt_rocblas_types.hpp"
#include "rocsolver_datatype2string.hpp"

/***************************************************************************
 * Wrapper for types passed to logger, so we can more easily adjust the
 * default way of printing built-in types without doing it globally. (e.g.
 * changing the printing of bool to "0" or "1" rather than "true" "false".)
 ***************************************************************************/
template <typename T>
struct rocsolver_logvalue
{
    T value;
};

/***************************************************************************
 * Constructs a rocsolver_logvalue given a value.
 * Used so that T can be inferred from the argument.
 ***************************************************************************/
template <typename T>
rocsolver_logvalue<T> rocsolver_make_logvalue(T value)
{
    return rocsolver_logvalue<T>{value};
}

namespace fmt
{
/* By default, forward log values to the original printer for their type. */
template <typename T>
struct formatter<rocsolver_logvalue<T>> : formatter<T>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<T> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<T>::format(wrapper.value, ctx);
    }
};

/* Specialize bool to print 0 or 1 rather than true or false, to match the
   rocsolver-bench CLI.*/
template <>
struct formatter<rocsolver_logvalue<bool>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<bool> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(wrapper.value ? '1' : '0', ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_operation>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_operation> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_operation(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_fill>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_fill> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_fill(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_diagonal>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_diagonal> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_diagonal(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_side>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_side> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_side(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_direct>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_direct> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_direct(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_storev>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_storev> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_storev(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_workmode>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_workmode> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_workmode(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_svect>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_svect> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_svect(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_srange>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_srange> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_srange(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_evect>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_evect> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_evect(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_eform>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_eform> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_eform(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_erange>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_erange> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_erange(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_eorder>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_eorder> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_eorder(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_esort>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_esort> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocblas2char_esort(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_datatype>> : formatter<string_view>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_datatype> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<string_view>::format(rocblas2string_datatype(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_initialization>> : formatter<string_view>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_initialization> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<string_view>::format(rocblas2string_initialization(wrapper.value), ctx);
    }
};

} // namespace
