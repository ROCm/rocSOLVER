/************************************************************************
 * Small sizes algorithm derived from MAGMA project
 * http://icl.cs.utk.edu/magma/.
 * https://doi.org/10.1016/j.procs.2017.05.250
 *
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#include "rocsolver_small_kernels.hpp"

#ifdef OPTIMAL

/************************************************************************
    LUfact_small_kernel takes care of of matrices with
    m <= GETF2_MAX_THDS and n <= GETF2_MAX_COLS
************************************************************************/
template <rocblas_int DIM, typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GETF2_MAX_THDS)
    LUfact_small_kernel(const rocblas_int m,
                        U AA,
                        const rocblas_int shiftA,
                        const rocblas_int lda,
                        const rocblas_stride strideA,
                        rocblas_int* ipivA,
                        const rocblas_int shiftP,
                        const rocblas_stride strideP,
                        rocblas_int* infoA,
                        const rocblas_int batch_count,
                        const int pivot)
{
    using S = decltype(std::real(T{}));

    int myrow = hipThreadIdx_x;
    const int ty = hipThreadIdx_y;
    const int id = hipBlockIdx_x * hipBlockDim_y + ty;

    if(id >= batch_count)
        return;

    // batch instance
    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    rocblas_int* ipiv;
    if(pivot)
        ipiv = load_ptr_batch<rocblas_int>(ipivA, id, shiftP, strideP);
    rocblas_int* info = infoA + id;

    // shared memory (for communication between threads in group)
    // (SHUFFLES DO NOT IMPROVE PERFORMANCE IN THIS CASE)
    extern __shared__ double lmem[];
    T* common = (T*)lmem;
    common += ty * GETF2_MAX_COLS;

    // local variables
    T pivot_value;
    T test_value;
    int pivot_index;
    int mypiv = myrow + 1; // to build ipiv
    int myinfo = 0; // to build info
    T rA[DIM]; // to store this-row values

    // read corresponding row from global memory into local array
#pragma unroll DIM
    for(int j = 0; j < DIM; ++j)
        rA[j] = A[myrow + j * lda];

        // for each pivot (main loop)
#pragma unroll DIM
    for(int k = 0; k < DIM; ++k)
    {
        // share current column
        common[myrow] = rA[k];
        __syncthreads();

        // search pivot index
        pivot_index = k;
        pivot_value = common[k];
        if(pivot)
        {
            for(int i = k + 1; i < m; ++i)
            {
                test_value = common[i];
                if(aabs<S>(pivot_value) < aabs<S>(test_value))
                {
                    pivot_value = test_value;
                    pivot_index = i;
                }
            }
        }

        // check singularity and scale value for current column
        if(pivot_value != T(0))
            pivot_value = S(1) / pivot_value;
        else if(myinfo == 0)
            myinfo = k + 1;

        // swap rows (lazy swaping)
        if(myrow == pivot_index)
        {
            myrow = k;
            // share pivot row
            for(int j = k + 1; j < DIM; ++j)
                common[j] = rA[j];
        }
        else if(myrow == k)
        {
            myrow = pivot_index;
            mypiv = pivot_index + 1;
        }
        __syncthreads();

        // scale current column and update trailing matrix
        if(myrow > k)
        {
            rA[k] *= pivot_value;
            for(int j = k + 1; j < DIM; ++j)
                rA[j] -= rA[k] * common[j];
        }
        __syncthreads();
    }

    // write results to global memory
    if(myrow < DIM && pivot)
        ipiv[myrow] = mypiv;
    if(myrow == 0)
        *info = myinfo;
#pragma unroll DIM
    for(int j = 0; j < DIM; ++j)
        A[myrow + j * lda] = rA[j];
}

/*************************************************************
    Launcher of LUfact kernels
*************************************************************/
template <typename T, typename U>
rocblas_status getf2_run_small(rocblas_handle handle,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               rocblas_int* ipiv,
                               const rocblas_int shiftP,
                               const rocblas_stride strideP,
                               rocblas_int* info,
                               const rocblas_int batch_count,
                               const rocblas_int pivot)
{
#define RUN_LUFACT_SMALL(DIM)                                                                      \
    hipLaunchKernelGGL((LUfact_small_kernel<DIM, T>), grid, block, lmemsize, stream, m, A, shiftA, \
                       lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot)

    // determine sizes
    int opval[] = {GETF2_OPTIM_NGRP};
    rocblas_int ngrp = (batch_count < 2 || m > 32) ? 1 : opval[m - 1];
    rocblas_int blocks = (batch_count - 1) / ngrp + 1;
    rocblas_int nthds = m;
    rocblas_int msize = (m <= 32) ? GETF2_MAX_COLS : max(m, n);

    // prepare kernel launch
    dim3 grid(blocks, 1, 1);
    dim3 block(nthds, ngrp, 1);
    size_t lmemsize = msize * ngrp * sizeof(T);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    // kernel launch
    switch(n)
    {
    case 1: RUN_LUFACT_SMALL(1); break;
    case 2: RUN_LUFACT_SMALL(2); break;
    case 3: RUN_LUFACT_SMALL(3); break;
    case 4: RUN_LUFACT_SMALL(4); break;
    case 5: RUN_LUFACT_SMALL(5); break;
    case 6: RUN_LUFACT_SMALL(6); break;
    case 7: RUN_LUFACT_SMALL(7); break;
    case 8: RUN_LUFACT_SMALL(8); break;
    case 9: RUN_LUFACT_SMALL(9); break;
    case 10: RUN_LUFACT_SMALL(10); break;
    case 11: RUN_LUFACT_SMALL(11); break;
    case 12: RUN_LUFACT_SMALL(12); break;
    case 13: RUN_LUFACT_SMALL(13); break;
    case 14: RUN_LUFACT_SMALL(14); break;
    case 15: RUN_LUFACT_SMALL(15); break;
    case 16: RUN_LUFACT_SMALL(16); break;
    case 17: RUN_LUFACT_SMALL(17); break;
    case 18: RUN_LUFACT_SMALL(18); break;
    case 19: RUN_LUFACT_SMALL(19); break;
    case 20: RUN_LUFACT_SMALL(20); break;
    case 21: RUN_LUFACT_SMALL(21); break;
    case 22: RUN_LUFACT_SMALL(22); break;
    case 23: RUN_LUFACT_SMALL(23); break;
    case 24: RUN_LUFACT_SMALL(24); break;
    case 25: RUN_LUFACT_SMALL(25); break;
    case 26: RUN_LUFACT_SMALL(26); break;
    case 27: RUN_LUFACT_SMALL(27); break;
    case 28: RUN_LUFACT_SMALL(28); break;
    case 29: RUN_LUFACT_SMALL(29); break;
    case 30: RUN_LUFACT_SMALL(30); break;
    case 31: RUN_LUFACT_SMALL(31); break;
    case 32: RUN_LUFACT_SMALL(32); break;
    case 33: RUN_LUFACT_SMALL(33); break;
    case 34: RUN_LUFACT_SMALL(34); break;
    case 35: RUN_LUFACT_SMALL(35); break;
    case 36: RUN_LUFACT_SMALL(36); break;
    case 37: RUN_LUFACT_SMALL(37); break;
    case 38: RUN_LUFACT_SMALL(38); break;
    case 39: RUN_LUFACT_SMALL(39); break;
    case 40: RUN_LUFACT_SMALL(40); break;
    case 41: RUN_LUFACT_SMALL(41); break;
    case 42: RUN_LUFACT_SMALL(42); break;
    case 43: RUN_LUFACT_SMALL(43); break;
    case 44: RUN_LUFACT_SMALL(44); break;
    case 45: RUN_LUFACT_SMALL(45); break;
    case 46: RUN_LUFACT_SMALL(46); break;
    case 47: RUN_LUFACT_SMALL(47); break;
    case 48: RUN_LUFACT_SMALL(48); break;
    case 49: RUN_LUFACT_SMALL(49); break;
    case 50: RUN_LUFACT_SMALL(50); break;
    case 51: RUN_LUFACT_SMALL(51); break;
    case 52: RUN_LUFACT_SMALL(52); break;
    case 53: RUN_LUFACT_SMALL(53); break;
    case 54: RUN_LUFACT_SMALL(54); break;
    case 55: RUN_LUFACT_SMALL(55); break;
    case 56: RUN_LUFACT_SMALL(56); break;
    case 57: RUN_LUFACT_SMALL(57); break;
    case 58: RUN_LUFACT_SMALL(58); break;
    case 59: RUN_LUFACT_SMALL(59); break;
    case 60: RUN_LUFACT_SMALL(60); break;
    case 61: RUN_LUFACT_SMALL(61); break;
    case 62: RUN_LUFACT_SMALL(62); break;
    case 63: RUN_LUFACT_SMALL(63); break;
    case 64: RUN_LUFACT_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }

    return rocblas_status_success;
}

/*************************************************************
    Instantiate template methods
*************************************************************/
// normal and strided_batched cases
template rocblas_status getf2_run_small<float, float*>(rocblas_handle,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       float*,
                                                       const rocblas_int,
                                                       const rocblas_int,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int,
                                                       const rocblas_stride,
                                                       rocblas_int*,
                                                       const rocblas_int,
                                                       const rocblas_int);

template rocblas_status getf2_run_small<double, double*>(rocblas_handle,
                                                         const rocblas_int,
                                                         const rocblas_int,
                                                         double*,
                                                         const rocblas_int,
                                                         const rocblas_int,
                                                         const rocblas_stride,
                                                         rocblas_int*,
                                                         const rocblas_int,
                                                         const rocblas_stride,
                                                         rocblas_int*,
                                                         const rocblas_int,
                                                         const rocblas_int);

template rocblas_status
    getf2_run_small<rocblas_float_complex, rocblas_float_complex*>(rocblas_handle,
                                                                   const rocblas_int,
                                                                   const rocblas_int,
                                                                   rocblas_float_complex*,
                                                                   const rocblas_int,
                                                                   const rocblas_int,
                                                                   const rocblas_stride,
                                                                   rocblas_int*,
                                                                   const rocblas_int,
                                                                   const rocblas_stride,
                                                                   rocblas_int*,
                                                                   const rocblas_int,
                                                                   const rocblas_int);

template rocblas_status
    getf2_run_small<rocblas_double_complex, rocblas_double_complex*>(rocblas_handle,
                                                                     const rocblas_int,
                                                                     const rocblas_int,
                                                                     rocblas_double_complex*,
                                                                     const rocblas_int,
                                                                     const rocblas_int,
                                                                     const rocblas_stride,
                                                                     rocblas_int*,
                                                                     const rocblas_int,
                                                                     const rocblas_stride,
                                                                     rocblas_int*,
                                                                     const rocblas_int,
                                                                     const rocblas_int);

// batched cases
template rocblas_status getf2_run_small<float, float* const*>(rocblas_handle,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              float* const*,
                                                              const rocblas_int,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int,
                                                              const rocblas_stride,
                                                              rocblas_int*,
                                                              const rocblas_int,
                                                              const rocblas_int);

template rocblas_status getf2_run_small<double, double* const*>(rocblas_handle,
                                                                const rocblas_int,
                                                                const rocblas_int,
                                                                double* const*,
                                                                const rocblas_int,
                                                                const rocblas_int,
                                                                const rocblas_stride,
                                                                rocblas_int*,
                                                                const rocblas_int,
                                                                const rocblas_stride,
                                                                rocblas_int*,
                                                                const rocblas_int,
                                                                const rocblas_int);

template rocblas_status getf2_run_small<rocblas_float_complex, rocblas_float_complex* const*>(
    rocblas_handle,
    const rocblas_int,
    const rocblas_int,
    rocblas_float_complex* const*,
    const rocblas_int,
    const rocblas_int,
    const rocblas_stride,
    rocblas_int*,
    const rocblas_int,
    const rocblas_stride,
    rocblas_int*,
    const rocblas_int,
    const rocblas_int);

template rocblas_status getf2_run_small<rocblas_double_complex, rocblas_double_complex* const*>(
    rocblas_handle,
    const rocblas_int,
    const rocblas_int,
    rocblas_double_complex* const*,
    const rocblas_int,
    const rocblas_int,
    const rocblas_stride,
    rocblas_int*,
    const rocblas_int,
    const rocblas_stride,
    rocblas_int*,
    const rocblas_int,
    const rocblas_int);

#endif // OPTIMAL
