/* **************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "rocauxiliary_orgqr_ungqr.hpp"

template <typename T>
rocblas_status rocsolver_orgqr_ungqr_impl(rocblas_handle handle,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int k,
                                          T* A,
                                          const rocblas_int lda,
                                          T* ipiv)
{
    const char* name = (!rocblas_is_complex<T> ? "orgqr" : "ungqr");
    ROCSOLVER_ENTER_TOP(name, "-m", m, "-n", n, "-k", k, "--lda", lda);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_org2r_orgqr_argCheck(handle, m, n, k, lda, A, ipiv);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORG2R/UNG2R and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orgqr_ungqr_getMemorySize<false, T>(m, n, k, batch_count, &size_scalars, &size_work,
                                                  &size_Abyx_tmptr, &size_trfact, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_work,
                                                      size_Abyx_tmptr, size_trfact, size_workArr);

    // memory workspace allocation
    void *scalars, *work, *Abyx_tmptr, *trfact, *workArr;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_Abyx_tmptr, size_trfact,
                              size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    Abyx_tmptr = mem[2];
    trfact = mem[3];
    workArr = mem[4];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_orgqr_ungqr_template<false, false, T>(
        handle, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, batch_count, (T*)scalars, (T*)work,
        (T*)Abyx_tmptr, (T*)trfact, (T**)workArr);
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sorgqr(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                float* A,
                                const rocblas_int lda,
                                float* ipiv)
{
    return rocsolver_orgqr_ungqr_impl<float>(handle, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_dorgqr(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                double* A,
                                const rocblas_int lda,
                                double* ipiv)
{
    return rocsolver_orgqr_ungqr_impl<double>(handle, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_cungqr(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_float_complex* A,
                                const rocblas_int lda,
                                rocblas_float_complex* ipiv)
{
    return rocsolver_orgqr_ungqr_impl<rocblas_float_complex>(handle, m, n, k, A, lda, ipiv);
}

rocblas_status rocsolver_zungqr(rocblas_handle handle,
                                const rocblas_int m,
                                const rocblas_int n,
                                const rocblas_int k,
                                rocblas_double_complex* A,
                                const rocblas_int lda,
                                rocblas_double_complex* ipiv)
{
    return rocsolver_orgqr_ungqr_impl<rocblas_double_complex>(handle, m, n, k, A, lda, ipiv);
}

} // extern C
