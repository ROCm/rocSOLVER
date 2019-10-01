/* ************************************************************************
 * Derived from the BSD2-licensed
 * LAPACK routine (version 3.8) --
 *     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
 *     December 2016
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCLAPACK_GEQRF_H
#define ROCLAPACK_GEQRF_H

#include <hip/hip_runtime.h>
#include "rocblas.hpp"
#include "rocsolver.h"
#include "definitions.h"
#include "helpers.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"
#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "../auxiliary/rocauxiliary_larf.hpp"

template <typename T, typename U>
rocblas_status rocsolver_geqrf_template(rocblas_handle handle, const rocblas_int m,
                                        const rocblas_int n, U A, const rocblas_int shiftA, const rocblas_int lda, 
                                        rocblas_int const strideA, T* ipiv, const rocblas_int shiftP, 
                                        const rocblas_int strideP, const rocblas_int batch_count)
{
    // quick return
    if (m == 0 || n == 0 || batch_count == 0) 
        return rocblas_status_success;
    
    // TO BE IMPLEMENTED...    

    return rocblas_status_success;
}

#endif /* ROCLAPACK_GEQRF_H */
