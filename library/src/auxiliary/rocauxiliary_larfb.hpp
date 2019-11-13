/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     November 2006
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_LARFB_HPP
#define ROCLAPACK_LARFB_HPP

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"

template <typename T, typename U>
rocblas_status rocsolver_larfb_template(rocsolver_handle handle, const rocsolver_side side, 
                                        const rocsolver_operation trans, const rocsolver_direct direct, 
                                        const rocsolver_int m, const rocsolver_int n,
                                        const rocsolver_int k, U V, const rocblas_int shiftV, const rocsolver_int ldv, 
                                        const rocsolver_int strideV, U F, const rocsolver_int shiftF,
                                        const rocsolver_int ldf, const rocsolver_int strideF, 
                                        U A, const rocsolver_int shiftA, const rocsolver_int lda, const rocsolver_int stridea,
                                        const rocsolver_int batch_count)
{
    // quick return
    if (!m || !n || !batch_count)
        return rocblas_status_success;

    //TO BE IMPLEMENTED....

    return rocblas_status_success;
}

#endif
