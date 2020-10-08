/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCSOLVER_DATATYPE2CHAR_H_
#define ROCSOLVER_DATATYPE2CHAR_H_

#include "rocblas.h"
#include "rocsolver.h"
#include <string>

/*  Convert rocblas constants to lapack char. */

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

/*  Convert lapack char constants to rocblas type. */

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

#endif
