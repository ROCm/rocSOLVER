/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "unit.h"
#include "rocsolver_test.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver.hpp"
#include "cblas_interface.h"
#include "clientcommon.hpp"


// this is max error PER element after the solution
#define GETRF_ERROR_EPS_MULTIPLIER 3000
// AS IN THE ORIGINAL ROCSOLVER TEST UNITS, WE CURRENTLY USE A HIGH TOLERANCE 
// AND THE MAX NORM TO EVALUATE THE ERROR. THIS IS NOT "NUMERICALLY SOUND"; 
// A MAJOR REFACTORING OF ALL UNIT TESTS WILL BE REQUIRED.  


// **** THIS FUNCTION ONLY TESTS NORMNAL USE CASE
//      I.E. WHEN STRIDEP >= M **** 

template <typename T>
void testing_getrs_batched_bad_arg()
{
    rocblas_local_handle handle;
    rocblas_int m = 1;
    rocblas_int nrhs = 1;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_operation trans = rocblas_operation_none;
    rocblas_int strideP = 1;
    rocblas_int batch_count = 3;

    device_batch_vector<T> dA(1,1,1);
    device_batch_vector<T> dB(1,1,1);
    device_strided_batch_vector<rocblas_int> dIpiv(1,1,1,1);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dIpiv.memcheck());

    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(nullptr, trans, m, nrhs, dA, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, rocblas_operation(-1), m, nrhs, dA, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, nullptr, lda, dIpiv, dB, ldb),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, nullptr, dB, ldb),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, nrhs, dA, lda, dIpiv, nullptr, ldb),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, 0, nrhs, nullptr, lda, nullptr, nullptr, ldb),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_getrs<T>(handle, trans, m, 0, dA, lda, dIpiv, nullptr, ldb),
                          rocblas_status_success);
}*/

template <typename T>
auto hostfunc(size_t size, rocblas_int inc, rocblas_int batch)
{
    host_batch_vector<T> queso(size,inc,batch);
    return queso;
}

template <typename T, typename U> 
void testing_getrs_batched(Arguments argus) 
{
    rocblas_local_handle handle;
    rocblas_int M = argus.M;
    rocblas_int nhrs = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_int ldb = argus.ldb;
    rocblas_int strideP = argus.bsp;
    char trans = argus.transA_option;
    int hot_calls = argus.iters;
    rocblas_int batch_count = argus.batch_count;

    rocblas_operation transRoc;
    if (trans == 'N') {
        transRoc = rocblas_operation_none;
    } else if (trans == 'T') {
        transRoc = rocblas_operation_transpose;
    } else if (trans == 'C') {
        transRoc = rocblas_operation_conjugate_transpose;
    } else {
        //throw runtime_error("Unsupported transpose operation.");
    }

    rocblas_int size_A = lda * M;
    rocblas_int size_B = ldb * nhrs;
    rocblas_int size_P = M;


    // check here to prevent undefined memory allocation error
    if (batch_count < 1 || M < 1 || nhrs < 1 || lda < M || ldb < M) {
<<<<<<< HEAD
        auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T*)), rocblas_test::device_free};
        T **dA = (T **)dA_managed.get();
=======
/*        T **dA;
        hipMalloc(&dA,sizeof(T*));
>>>>>>> re-use rocblas-clients' device_ and host_ vectors

        auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(T*)), rocblas_test::device_free};
        T **dB = (T **)dB_managed.get();

        auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int)), rocblas_test::device_free};
        rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();

        if (!dA || !dIpiv || !dB) {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return rocblas_status_memory_error;
        }
*/
//        return rocsolver_getrs_batched<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, strideP, dB, ldb, batch_count);
        return;
    }

//    size_P += strideP * (batch_count - 1);

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
/*    vector<T> hA[batch_count];
    vector<T> hB[batch_count];
    vector<int> hIpiv(size_P);
    vector<T> hBRes[batch_count];
    for(int b=0; b < batch_count; ++b) {
        hA[b] = vector<T>(size_A);
        hB[b] = vector<T>(size_B);
        hBRes[b] = vector<T>(size_B);
    }        
*/

//    host_batch_vector<T> hA(size_A,1,batch_count);
//    host_batch_vector<T> hB(size_B,1,batch_count);
//    host_batch_vector<T> hBRes(size_B,1,batch_count);
    host_strided_batch_vector<rocblas_int> hIpiv(size_P,1,strideP,batch_count);
        
    device_batch_vector<T> dA(size_A,1,batch_count);
    device_batch_vector<T> dB(size_B,1,batch_count);
    device_strided_batch_vector<rocblas_int> dIpiv(size_P,1,strideP,batch_count);

    double gpu_time_used, cpu_time_used;
    double error_eps_multiplier = GETRF_ERROR_EPS_MULTIPLIER;
    double eps = std::numeric_limits<U>::epsilon();

    // allocate memory on device
/*    T* A[batch_count];
    T* B[batch_count];
    for(int b=0; b < batch_count; ++b) {
        hipMalloc(&A[b], sizeof(T) * size_A);
        hipMalloc(&B[b], sizeof(T) * size_B);
    }
    T **dA, **dB;
    hipMalloc(&dA,sizeof(T*) * batch_count);
    hipMalloc(&dB,sizeof(T*) * batch_count);
    auto dIpiv_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(int) * size_P), rocblas_test::device_free};
    rocblas_int *dIpiv = (rocblas_int *)dIpiv_managed.get();
  
    if (!dA || !dIpiv || !dB || !A[batch_count-1] || !B[batch_count-1]) {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return rocblas_status_memory_error;
    }
*/

    rocblas_init<T>(hA,false);
    rocblas_cout << std::endl;
    rocblas_cout << hA[0][0] << " " << hA[1][0] << " " << hA[2][0];    

    //  initialize full random matrix h and hB 
    for(int b=0; b < batch_count; ++b) {
//        rocblas_init<T>(hA[b], M, M, lda);
        rocblas_init<T>(hB[b], M, nhrs, ldb);



        // put it into [0, 1]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < M; j++) {
                if (i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }
    }

    // do the LU decomposition of matrix A w/ the reference LAPACK routine
    int retCBLAS;
    for(int b=0; b < batch_count; ++b) {
        retCBLAS = 0;
        cblas_getrf<T>(M, M, hA[b], lda, hIpiv[b], &retCBLAS);
        if (retCBLAS != 0) {
            // error encountered - unlucky pick of random numbers? no use to continue
            return;
        }
    }

    // now copy pivoting indices and matrices to the GPU
/*    for(int b=0;b<batch_count;b++) {
        CHECK_HIP_ERROR(hipMemcpy(A[b], hA[b].data(), sizeof(T)*size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(B[b], hB[b].data(), sizeof(T)*size_B, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dA, A, sizeof(T*)*batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, B, sizeof(T*)*batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), sizeof(int) * size_P, hipMemcpyHostToDevice));
*/

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));

    double max_err_1 = 0.0, max_val = 0.0, diff, err;

/* =====================================================================
           ROCSOLVER
    =================================================================== */
    if (argus.unit_check || argus.norm_check) {
        //GPU lapack
        CHECK_ROCBLAS_ERROR(rocsolver_getrs_batched<T>(handle, transRoc, M, nhrs, dA.ptr_on_device(), lda, dIpiv, strideP, dB.ptr_on_device(), ldb, batch_count));
        CHECK_HIP_ERROR(hBRes.transfer_from(dB));
//        for(int b=0;b<batch_count;b++) 
//            CHECK_HIP_ERROR(hipMemcpy(hBRes[b].data(), B[b], sizeof(T) * size_B, hipMemcpyDeviceToHost));

        //CPU lapack
        cpu_time_used = get_time_us();
        for(int b=0; b < batch_count; ++b) 
            cblas_getrs<T>(trans, M, nhrs, hA[b], lda, hIpiv[b], hB[b], ldb);
        cpu_time_used = get_time_us() - cpu_time_used;


        // Error Check
        for(int b=0; b < batch_count; ++b) {
            err = 0.0;
            max_val = 0.0;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < nhrs; j++) {
                    diff = std::abs(hB[b][i + j * ldb]);
                    max_val = max_val > diff ? max_val : diff;
                    diff = std::abs(hBRes[b][i + j * ldb] - hB[b][i + j * ldb]);
                    err = err > diff ? err : diff;
                }
            }
            err = err / max_val;
            max_err_1 = max_err_1 > err ? max_err_1 : err;
        }

        getrs_err_res_check<U>(max_err_1, M, nhrs, error_eps_multiplier, eps);
    }

    if (argus.timing) {
        // GPU rocBLAS
        int cold_calls = 2;

        for(int iter = 0; iter < cold_calls; iter++)
            rocsolver_getrs_batched<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, strideP, dB, ldb, batch_count);
        gpu_time_used = get_time_us(); // in microseconds
        for(int iter = 0; iter < hot_calls; iter++)
            rocsolver_getrs_batched<T>(handle, transRoc, M, nhrs, dA, lda, dIpiv, strideP, dB, ldb, batch_count);
        gpu_time_used = get_time_us() - gpu_time_used;

        // only norm_check return an norm error, unit check won't return anything
        rocblas_cout << "trans , M , nhrs , lda , strideP , ldb , batch_count , us [gpu] , us [cpu]";

        if (argus.norm_check)
            rocblas_cout << ", norm_error_host_ptr";

        rocblas_cout << std::endl;

        rocblas_cout << trans << " , " << M << " , " << nhrs << " , " << lda << " , " << strideP << " , " << ldb << " , " << batch_count << " , " << gpu_time_used << " , " << cpu_time_used;

        if (argus.norm_check)
            rocblas_cout << " , " << max_err_1;

        rocblas_cout << std::endl;
    }

//    for(int b=0;b<batch_count;++b) {
//        hipFree(A[b]);
//        hipFree(B[b]);
//    }
//    hipFree(dA);
//    hipFree(dB);
    
}

#undef GETRF_ERROR_EPS_MULTIPLIER
