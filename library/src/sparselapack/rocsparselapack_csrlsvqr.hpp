/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "../lapack/roclapack_gels.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_spinfo.hpp"
#include "rocsparse.hpp"

#include "common_host_helpers.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_csrlsvqr_impl(rocblas_handle handle,
                                       const rocblas_int n,
                                       const rocblas_int nnz,
                                       U A,
                                       rocblas_int* ptrA,
                                       rocblas_int* indA,
                                       U B,
                                       const T tol,
                                       const rocblas_int reorder,
                                       T* X,
                                       int* singularity,
                                       rocsolver_spinfo spinfo)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    ROCSOLVER_ENTER_TOP("csrlsvqr", "-n", n, "--nnz", nnz, "--tol", tol);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_device_malloc sparse_mem(handle, sizeof(T) * n * n, sizeof(T) * n, sizeof(rocblas_int));

    // scatter to dense matrix
    void* denseA = sparse_mem[0];
    void* bCopy = sparse_mem[1];

    rocsparseCall_csr2dense(spinfo->sphandle, n, n, spinfo->descrA, A, ptrA, indA, (T*)denseA, n);

    // one B at a time, square matrices only
    const rocblas_int lda = n;
    const rocblas_int ldb = n;
    const rocblas_int nrhs = 1;
    const rocblas_operation trans = rocblas_operation_none;

    // working with unshifted arrays
    const rocblas_int shiftA = 0;
    const rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    const rocblas_stride strideA = 0;
    const rocblas_stride strideB = 0;
    const rocblas_int batch_count = 1;

    rocblas_int* info = (rocblas_int*)sparse_mem[2];

    // argument checking
    rocblas_status st
        = rocsolver_gels_argCheck<COMPLEX>(handle, trans, n, n, nrhs, (T*)denseA, lda, B, ldb, info);
    if(st != rocblas_status_continue)
        return st;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of workspace (for calling GEQRF/GELQF, ORMQR/ORMLQ, and TRSM)
    bool optim_mem;
    size_t size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
        size_trfact_workTrmm_invA_arr;
    // extra requirements for calling ORMQR/ORMLQ and to copy B
    size_t size_ipiv_savedB;
    rocsolver_gels_getMemorySize<false, false, T>(
        trans, n, n, nrhs, batch_count, &size_scalars, &size_work_x_temp, &size_workArr_temp_arr,
        &size_diag_trfac_invA, &size_trfact_workTrmm_invA_arr, &size_ipiv_savedB, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work_x_temp, size_workArr_temp_arr, size_diag_trfac_invA,
            size_trfact_workTrmm_invA_arr, size_ipiv_savedB);

    // memory workspace allocation
    void *scalars, *work_x_temp, *workArr_temp_arr, *diag_trfac_invA, *trfact_workTrmm_invA_arr,
        *ipiv_savedB;
    rocblas_device_malloc mem(handle, size_scalars, size_work_x_temp, size_workArr_temp_arr,
                              size_diag_trfac_invA, size_trfact_workTrmm_invA_arr, size_ipiv_savedB);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work_x_temp = mem[1];
    workArr_temp_arr = mem[2];
    diag_trfac_invA = mem[3];
    trfact_workTrmm_invA_arr = mem[4];
    ipiv_savedB = mem[5];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*>), dim3(n, 1, 1), dim3(32, 1, 1), 0, stream, n, 1, B,
                            shiftB, batch_count, strideB, (T*)bCopy, shiftB, batch_count, strideB);

    st = rocsolver_gels_template<false, false, T>(
        handle, trans, n, n, nrhs, (T*)denseA, shiftA, lda, strideA, (T*)bCopy, shiftB, ldb,
        strideB, info, batch_count, (T*)scalars, (T*)work_x_temp, (T*)workArr_temp_arr,
        (T*)diag_trfac_invA, (T**)trfact_workTrmm_invA_arr, (T*)ipiv_savedB, optim_mem);

    if(st == rocblas_status_success)
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*>), dim3(n, 1, 1), dim3(32, 1, 1), 0, stream, n, 1,
                                (T*)bCopy, shiftB, batch_count, strideB, X, shiftB, batch_count,
                                strideB);

    print_device_matrix(std::cout, "X", n, 1, X, 1);

    return st;
}

ROCSOLVER_END_NAMESPACE
