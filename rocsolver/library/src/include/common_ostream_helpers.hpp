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
