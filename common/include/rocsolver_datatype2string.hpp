/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocsolver.h"
#include <string>

typedef enum rocblas_initialization_ : int
{
    rocblas_initialization_random_int = 111,
    rocblas_initialization_trig_float = 222,
    rocblas_initialization_hpl = 333,
} rocblas_initialization;

// return char from type
template <typename>
static constexpr char rocblas2char_precision = '\0';
template <>
static constexpr auto rocblas2char_precision<float> = 's';
template <>
static constexpr auto rocblas2char_precision<double> = 'd';
template <>
static constexpr auto rocblas2char_precision<rocblas_float_complex> = 'c';
template <>
static constexpr auto rocblas2char_precision<rocblas_double_complex> = 'z';

/* ============================================================================================
 */
/*  Convert rocblas constants to lapack char. */

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none: return 'N';
    case rocblas_operation_transpose: return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full: return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit: return 'U';
    case rocblas_diagonal_non_unit: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left: return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both: return 'B';
    }
    return '\0';
}

constexpr auto rocblas2char_direct(rocblas_direct value)
{
    switch(value)
    {
    case rocblas_forward_direction: return 'F';
    case rocblas_backward_direction: return 'B';
    }
    return '\0';
}

constexpr auto rocblas2char_storev(rocblas_storev value)
{
    switch(value)
    {
    case rocblas_column_wise: return 'C';
    case rocblas_row_wise: return 'R';
    }
    return '\0';
}

constexpr auto rocblas2char_workmode(rocblas_workmode value)
{
    switch(value)
    {
    case rocblas_outofplace: return 'O';
    case rocblas_inplace: return 'I';
    }
    return '\0';
}

constexpr auto rocblas2char_svect(rocblas_svect value)
{
    switch(value)
    {
    case rocblas_svect_all: return 'A';
    case rocblas_svect_singular: return 'S';
    case rocblas_svect_overwrite: return 'O';
    case rocblas_svect_none: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_evect(rocblas_evect value)
{
    switch(value)
    {
    case rocblas_evect_original: return 'V';
    case rocblas_evect_tridiagonal: return 'I';
    case rocblas_evect_none: return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_eform(rocblas_eform value)
{
    switch(value)
    {
    case rocblas_eform_ax: return '1';
    case rocblas_eform_abx: return '2';
    case rocblas_eform_bax: return '3';
    }
    return '\0';
}

constexpr auto rocblas2char_eval_range(rocblas_eval_range value)
{
    switch(value)
    {
    case rocblas_range_all: return 'A';
    case rocblas_range_value: return 'V';
    case rocblas_range_index: return 'I';
    }
    return '\0';
}

constexpr auto rocblas2char_eval_order(rocblas_eval_order value)
{
    switch(value)
    {
    case rocblas_order_blocks: return 'B';
    case rocblas_order_entire: return 'E';
    }
    return '\0';
}

// return precision string for rocblas_datatype
constexpr auto rocblas2string_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r: return "f16_r";
    case rocblas_datatype_f32_r: return "f32_r";
    case rocblas_datatype_f64_r: return "f64_r";
    case rocblas_datatype_f16_c: return "f16_k";
    case rocblas_datatype_f32_c: return "f32_c";
    case rocblas_datatype_f64_c: return "f64_c";
    case rocblas_datatype_i8_r: return "i8_r";
    case rocblas_datatype_u8_r: return "u8_r";
    case rocblas_datatype_i32_r: return "i32_r";
    case rocblas_datatype_u32_r: return "u32_r";
    case rocblas_datatype_i8_c: return "i8_c";
    case rocblas_datatype_u8_c: return "u8_c";
    case rocblas_datatype_i32_c: return "i32_c";
    case rocblas_datatype_u32_c: return "u32_c";
    case rocblas_datatype_bf16_r: return "bf16_r";
    case rocblas_datatype_bf16_c: return "bf16_c";
    }
    return "invalid";
}

constexpr auto rocblas2string_initialization(rocblas_initialization init)
{
    switch(init)
    {
    case rocblas_initialization_random_int: return "rand_int";
    case rocblas_initialization_trig_float: return "trig_float";
    case rocblas_initialization_hpl: return "hpl";
    }
    return "invalid";
}

/* ============================================================================================
 */
/*  Convert lapack char constants to rocblas type. */

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n': return rocblas_operation_none;
    case 'T':
    case 't': return rocblas_operation_transpose;
    case 'C':
    case 'c': return rocblas_operation_conjugate_transpose;
    default: return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_fill_upper;
    case 'L':
    case 'l': return rocblas_fill_lower;
    default: return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u': return rocblas_diagonal_unit;
    case 'N':
    case 'n': return rocblas_diagonal_non_unit;
    default: return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l': return rocblas_side_left;
    case 'R':
    case 'r': return rocblas_side_right;
    default: return static_cast<rocblas_side>(-1);
    }
}

constexpr rocblas_direct char2rocblas_direct(char value)
{
    switch(value)
    {
    case 'F': return rocblas_forward_direction;
    case 'B': return rocblas_backward_direction;
    default: return static_cast<rocblas_direct>(-1);
    }
}

constexpr rocblas_storev char2rocblas_storev(char value)
{
    switch(value)
    {
    case 'C': return rocblas_column_wise;
    case 'R': return rocblas_row_wise;
    default: return static_cast<rocblas_storev>(-1);
    }
}

constexpr rocblas_workmode char2rocblas_workmode(char value)
{
    switch(value)
    {
    case 'O': return rocblas_outofplace;
    case 'I': return rocblas_inplace;
    default: return static_cast<rocblas_workmode>(-1);
    }
}

constexpr rocblas_svect char2rocblas_svect(char value)
{
    switch(value)
    {
    case 'A': return rocblas_svect_all;
    case 'S': return rocblas_svect_singular;
    case 'O': return rocblas_svect_overwrite;
    case 'N': return rocblas_svect_none;
    default: return static_cast<rocblas_svect>(-1);
    }
}

constexpr rocblas_evect char2rocblas_evect(char value)
{
    switch(value)
    {
    case 'V': return rocblas_evect_original;
    case 'I': return rocblas_evect_tridiagonal;
    case 'N': return rocblas_evect_none;
    default: return static_cast<rocblas_evect>(-1);
    }
}

constexpr rocblas_eform char2rocblas_eform(char value)
{
    switch(value)
    {
    case '1': return rocblas_eform_ax;
    case '2': return rocblas_eform_abx;
    case '3': return rocblas_eform_bax;
    default: return static_cast<rocblas_eform>(-1);
    }
}

constexpr rocblas_eval_range char2rocblas_eval_range(char value)
{
    switch(value)
    {
    case 'A': return rocblas_range_all;
    case 'V': return rocblas_range_value;
    case 'I': return rocblas_range_index;
    default: return static_cast<rocblas_eval_range>(-1);
    }
}

constexpr rocblas_eval_order char2rocblas_eval_order(char value)
{
    switch(value)
    {
    case 'B': return rocblas_order_blocks;
    case 'E': return rocblas_order_entire;
    default: return static_cast<rocblas_eval_order>(-1);
    }
}

// clang-format off
inline rocblas_initialization string2rocblas_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? rocblas_initialization_random_int :
        value == "trig_float" ? rocblas_initialization_trig_float :
        value == "hpl"        ? rocblas_initialization_hpl        :
        static_cast<rocblas_initialization>(-1);
}

inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r  :
        value == "i32_r"                 ? rocblas_datatype_i32_r :
        value == "i8_c"                  ? rocblas_datatype_i8_c  :
        value == "i32_c"                 ? rocblas_datatype_i32_c :
        value == "u8_r"                  ? rocblas_datatype_u8_r  :
        value == "u32_r"                 ? rocblas_datatype_u32_r :
        value == "u8_c"                  ? rocblas_datatype_u8_c  :
        value == "u32_c"                 ? rocblas_datatype_u32_c :
        static_cast<rocblas_datatype>(-1);
}
// clang-format on
