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
                                       int m,
                                       int nnz,
                                       const rocsolver_rfinfo rfinfo,
                                       const T* A,
                                       const int* csrRowPtrA,
                                       const int* csrColIndA,
                                       const T* b,
                                       T tol,
                                       int reorder,
                                       T* x,
                                       int* singularity)
{
    return rocblas_status_not_implemented;
}

ROCSOLVER_END_NAMESPACE
