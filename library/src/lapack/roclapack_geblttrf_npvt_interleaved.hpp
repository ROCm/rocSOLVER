/************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#define GEBLT_BLOCK_DIM 256
#define NB_SMALL 16

#define indx4f(i1, i2, i3, i4, n1, n2, n3) \
    (indx3f(i1, i2, i3, n1, n2) + ((i4)-1) * (((int64_t)(n1)) * (n2)) * (n3))
#define indx3f(i1, i2, i3, n1, n2) (indx2f(i1, i2, n1) + ((i3)-1) * (((int64_t)(n1)) * (n2)))
#define indx2f(i1, i2, n1) (((i1)-1) + ((i2)-1) * ((int64_t)(n1)))

/*
! ------------------------------------------------------
!     Perform LU factorization without pivoting
!     of block tridiagonal matrix
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A1, B2, C2     ] = [ A1 D2      ] * [    I  U2    ]
! % [    A2, B3, C3 ]   [    A2 D3   ]   [       I  U3 ]
! % [        A3, B4 ]   [       A3 D4]   [          I4 ]
! ------------------------------------------------------
*/

template <typename T>
__device__ void gemm_nn_bf_device(const rocblas_int batch_count,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  const rocblas_int k,
                                  const T alpha,
                                  T* A_,
                                  const rocblas_int lda,
                                  T* B_,
                                  const rocblas_int ldb,
                                  const T beta,
                                  T* C_,
                                  const rocblas_int ldc)
{
#define A(iv, ia, ja) A_[indx3f(iv, ia, ja, batch_count, lda)]
#define B(iv, ib, jb) B_[indx3f(iv, ib, jb, batch_count, ldb)]
#define C(iv, ic, jc) C_[indx3f(iv, ic, jc, batch_count, ldc)]

    rocblas_int const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    rocblas_int const iv_end = batch_count;
    rocblas_int const iv_inc = (gridDim.x * blockDim.x);

    T const zero = 0;

    bool const is_beta_zero = (beta == zero);

    for(rocblas_int jc = 1; jc <= n; jc++)
    {
        for(rocblas_int ic = 1; ic <= m; ic++)
        {
            for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
            {
                T cij = zero;
                for(rocblas_int ja = 1; ja <= k; ja++)
                {
                    cij += A(iv, ic, ja) * B(iv, ja, jc);
                };

                if(is_beta_zero)
                {
                    C(iv, ic, jc) = alpha * cij;
                }
                else
                {
                    C(iv, ic, jc) = beta * C(iv, ic, jc) + alpha * cij;
                };
            }; // end for iv
            __syncthreads();

        }; // end for ic
    }; // end for jc

#undef A
#undef B
#undef C
}

template <typename T, typename I>
__device__ void
    getrf_npvt_bf_device(I const batchCount, I const m, I const n, T* A_, I const lda, I info[])
{
    I const min_mn = (m < n) ? m : n;
    T const one = 1;

    I const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    I const iv_end = batchCount;
    I const iv_inc = (gridDim.x * blockDim.x);

#define A(iv, i, j) A_[indx3f(iv, i, j, batchCount, lda)]

    T const zero = 0;

    for(I j = 1; j <= min_mn; j++)
    {
        I const jp1 = j + 1;

        for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
        {
            bool const is_diag_zero = (std::abs(A(iv, j, j)) == zero);
            T const Ujj_iv = is_diag_zero ? one : A(iv, j, j);
            info[iv - 1] = is_diag_zero && (info[iv - 1] == 0) ? j : info[iv - 1];

            for(I ia = jp1; ia <= m; ia++)
            {
                A(iv, ia, j) = A(iv, ia, j) / Ujj_iv;
            };
        };
        __syncthreads();

        for(I ja = jp1; ja <= n; ja++)
        {
            for(I ia = jp1; ia <= m; ia++)
            {
                for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    A(iv, ia, ja) = A(iv, ia, ja) - A(iv, ia, j) * A(iv, j, ja);
                };
            };
        };
        __syncthreads();
    };

#undef A
}

template <typename T>
__device__ void getrs_npvt_bf(rocblas_int const batchCount,
                              rocblas_int const n,
                              rocblas_int const nrhs,
                              T* A_,
                              rocblas_int const lda,
                              T* B_,
                              rocblas_int const ldb,
                              rocblas_int* pinfo)
{
    /*
    !     ---------------------------------------------------
    !     Perform forward and backward solve without pivoting
    !     ---------------------------------------------------
    */

#define A(iv, ia, ja) A_[indx3f(iv, ia, ja, batchCount, lda)]
#define B(iv, ib, irhs) B_[indx3f(iv, ib, irhs, batchCount, ldb)]

    rocblas_int const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = (gridDim.x * blockDim.x);

    T const one = 1;
    T const zero = 0;

    rocblas_int info = 0;
    /*
    !
    ! % ------------------------
    ! % L * (U * X) = B
    ! % step 1: solve L * Y = B
    ! % step 2: solve U * X = Y
    ! % ------------------------
    !
    !
    ! % ------------------------------
    ! % [I         ] [ Y1 ]   [ B1 ]
    ! % [L21 I     ] [ Y2 ] = [ B2 ]
    ! % [L31 L21 I ] [ Y3 ]   [ B3 ]
    ! % ------------------------------
    !
    !
    ! % ------------
    ! % special case
    ! % ------------
    */

    for(rocblas_int i = 1; i <= n; i++)
    {
        for(rocblas_int j = 1; j <= (i - 1); j++)
        {
            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    B(iv, i, k) = B(iv, i, k) - A(iv, i, j) * B(iv, j, k);
                };
            };
            __syncthreads();
        };
    };
    /*
    ! % ------------------------------
    ! % [U11 U12 U13 ] [ X1 ] = [ Y1 ]
    ! % [    U22 U23 ]*[ X2 ] = [ Y2 ]
    ! % [        U33 ]*[ X3 ] = [ Y3 ]
    ! % ------------------------------
    */
    for(rocblas_int ir = 1; ir <= n; ir++)
    {
        rocblas_int i = n - ir + 1;
        for(rocblas_int j = (i + 1); j <= n; j++)
        {
            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    B(iv, i, k) = B(iv, i, k) - A(iv, i, j) * B(iv, j, k);
                };
            };
            __syncthreads();

        }; // end for j

        for(rocblas_int iv = 1; iv <= iv_end; iv += iv_inc)
        {
            T const A_iv_i_i = A(iv, i, i);
            bool const is_diag_zero = (std::abs(A_iv_i_i) == zero);
            info = (is_diag_zero && (info == 0)) ? i : info;

            T const inv_Uii_iv = (is_diag_zero) ? one : one / A_iv_i_i;

            for(rocblas_int k = 1; k <= nrhs; k++)
            {
                B(iv, i, k) = B(iv, i, k) * inv_Uii_iv;
            };

        }; // end for iv
        __syncthreads();

    }; // end for ir

    *pinfo = info;

#undef A
#undef B
}

template <typename T, typename I>
__global__ __launch_bounds__(GEBLT_BLOCK_DIM) void geblttrf_npvt_bf_kernel(I const nb,
                                                                           I const nblocks,
                                                                           T* A_,
                                                                           I const lda,
                                                                           T* B_,
                                                                           I const ldb,
                                                                           T* C_,
                                                                           I const ldc,
                                                                           I devinfo_array[],
                                                                           I batch_count)
{
// note adjust indexing for array A
#define A(iv, ia, ja, k) A_[indx4f(iv, ia, ja, ((k)-1), batch_count, lda, nb)]

#define B(iv, ib, jb, k) B_[indx4f(iv, ib, jb, k, batch_count, ldb, nb)]
#define C(iv, ic, jc, k) C_[indx4f(iv, ic, jc, k, batch_count, ldc, nb)]

#define D(iv, i, j, k) B(iv, i, j, k)
#define U(iv, i, j, k) C(iv, i, j, k)
    I const ldu = ldc;
    I const ldd = ldb;
    /*
    !
    ! % B1 = D1
    ! % D1 * U1 = C1 => U1 = D1 \ C1
    ! % D2 + A2*U1 = B2 => D2 = B2 - A2*U1
    ! %
    ! % D2*U2 = C2 => U2 = D2 \ C2
    ! % D3 + A3*U2 = B3 => D3 = B3 - A3*U2
    ! %
    ! % D3*U3 = C3 => U3 = D3 \ C3
    ! % D4 + A4*U3 = B4 => D4 = B4 - A4*U3
    ! idebug = 1;
    !
    ! % ----------------------------------
    ! % in actual code, overwrite B with D
    ! % overwrite C with U
    ! % ----------------------------------
    */

    {
        I const iv = 1;
        I const k = 1;
        I const mm = nb;
        I const nn = nb;
        T* Ap = &(D(iv, 1, 1, k));

        getrf_npvt_bf_device<T>(batch_count, mm, nn, Ap, ldd, devinfo_array);
        __syncthreads();
    };

    for(I k = 1; k <= (nblocks - 1); k++)
    {
        {
            I const nn = nb;
            I const nrhs = nb;
            I const iv = 1;

            T* Ap = &(D(iv, 1, 1, k));
            T* Bp = &(C(iv, 1, 1, k));
            getrs_npvt_bf<T>(batch_count, nn, nrhs, Ap, ldd, Bp, ldc, devinfo_array);
            __syncthreads();
        };

        {
            I const iv = 1;
            I const mm = nb;
            I const nn = nb;
            I const kk = nb;
            T const alpha = -1;
            T const beta = 1;
            I const ld1 = lda;
            I const ld2 = ldu;
            I const ld3 = ldd;

            T* Ap = &(A(iv, 1, 1, k + 1));
            T* Bp = &(U(iv, 1, 1, k));
            T* Cp = &(D(iv, 1, 1, k + 1));
            gemm_nn_bf_device<T>(batch_count, mm, nn, kk, alpha, Ap, ld1, Bp, ld2, beta, Cp, ld3);
            __syncthreads();
        };

        {
            I const iv = 1;
            I const mm = nb;
            I const nn = nb;
            T* Ap = &(D(iv, 1, 1, k + 1));

            getrf_npvt_bf_device<T>(batch_count, mm, nn, Ap, ldd, devinfo_array);
            __syncthreads();
        };

    }; // end for k

#undef D
#undef U

#undef A
#undef B
#undef C
}

template <typename T>
void rocsolver_geblttrf_npvt_interleaved_getMemorySize(const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int batch_count,
                                                       size_t* size_work)
{
    // if quick return, no need of workspace
    if(nb == 0 || nblocks == 0 || batch_count == 0)
    {
        // TODO: set workspace sizes to zero
        *size_work = 0;
        return;
    }

    // TODO: calculate workspace sizes
    *size_work = 0;
}

template <typename T>
rocblas_status rocsolver_geblttrf_npvt_interleaved_argCheck(rocblas_handle handle,
                                                            const rocblas_int nb,
                                                            const rocblas_int nblocks,
                                                            const rocblas_int lda,
                                                            const rocblas_int ldb,
                                                            const rocblas_int ldc,
                                                            T A,
                                                            T B,
                                                            T C,
                                                            rocblas_int* info,
                                                            const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    };

    // 2. invalid size
    if(nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((nb && nblocks > 1 && !A) || (nb && nblocks && !B) || (nb && nblocks > 1 && !C)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_geblttrf_npvt_interleaved_template(rocblas_handle handle,
                                                            const rocblas_int nb,
                                                            const rocblas_int nblocks,
                                                            U A,
                                                            const rocblas_int lda,
                                                            U B,
                                                            const rocblas_int ldb,
                                                            U C,
                                                            const rocblas_int ldc,
                                                            rocblas_int* info,
                                                            const rocblas_int batch_count,
                                                            void* work)
{
    ROCSOLVER_ENTER("geblttrf_npvt_interleaved", "nb:", nb, "nblocks:", nblocks, "lda:", lda,
                    "ldb:", ldb, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(nb == 0 || nblocks == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threadsReset(BS1, 1, 1);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

    auto const block_dim = GEBLT_BLOCK_DIM;
    auto const grid_dim = (batch_count + (block_dim - 1)) / block_dim;
    ROCSOLVER_LAUNCH_KERNEL(geblttrf_npvt_bf_kernel, dim3(grid_dim), dim3(block_dim), 0, stream, nb,
                            nblocks, A, lda, B, ldb, C, ldc, info, batch_count);

    return rocblas_status_success;
}
