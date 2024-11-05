/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_csrlsvqr_impl(rocblas_handle handle,
                                       const rocblas_int m,
                                       const rocblas_int nnz,
                                       const rocsparse_mat_descr descA,
                                       const T* A,
                                       const rocblas_int* ptrA,
                                       const rocblas_int* indA,
                                       const T* b,
                                       const T tol,
                                       const rocblas_int reorder,
                                       T* x,
                                       int* singularity)
{
    return rocblas_status_not_implemented;
}

ROCSOLVER_END_NAMESPACE
