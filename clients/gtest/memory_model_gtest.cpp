/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>

#include <gtest/gtest.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "client_environment_helpers.hpp"

class checkin_misc_MEMORY_MODEL : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(char* envvar = getenv("ROCBLAS_DEVICE_MEMORY_SIZE"))
            GTEST_SKIP() << "Cannot execute in dirty environment; ROCBLAS_DEVICE_MEMORY_SIZE="
                         << envvar;

        ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * bc), hipSuccess);
    }

    void TearDown() override
    {
        if(getenv("ROCBLAS_DEVICE_MEMORY_SIZE"))
            unset_environment_variable("ROCBLAS_DEVICE_MEMORY_SIZE");

        ASSERT_EQ(hipFree(dA), hipSuccess);
        ASSERT_EQ(hipFree(dP), hipSuccess);
        ASSERT_EQ(hipFree(dinfo), hipSuccess);
    }

    double* dA;
    rocblas_int *dP, *dinfo;

    const rocblas_int m = 1500;
    const rocblas_int n = 1500;
    const rocblas_int m_small = 65;
    const rocblas_int n_small = 65;
    const rocblas_int lda = m;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;
    const rocblas_int bc = 8;
    const rocblas_int bc_small = 5;
};

/*************************************/
/***** rocblas_managed (default) *****/
/*************************************/
TEST_F(checkin_misc_MEMORY_MODEL, rocblas_managed)
{
    size_t size, size1;
    rocblas_status status;
    rocblas_handle handle;

    // 1. create handle
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    // 2. by default, memory is rocblas managed
    EXPECT_TRUE(rocblas_is_managing_device_memory(handle));

    // 3. by default, 32MB should be reserved
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 32 * 1024 * 1024);

    // 4. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 5. getrf baseline will require ~54MB
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 6. stop query
    rocblas_stop_device_memory_size_query(handle, &size1);
    EXPECT_GT(size1, 32 * 1024 * 1024);

    // 7. device memory size should not change yet; it should be 32MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 32 * 1024 * 1024);

    // 8. When executing getrf, rocblas should increase memory automatically
    // allowing execution to success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_success);

    // 9. device memory size should have changed after execution of getrf to 54MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, size1);

    // 10. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 11. getrf small will require ~.5MB
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 12. stop query
    rocblas_stop_device_memory_size_query(handle, &size);
    EXPECT_LT(size, size1);

    // 13. device memory size should not change; it should be 54MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, size1);

    // 14. When executing getrf, device memory is enough for execution to success
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_success);

    // 15. device memory size should be the same 54MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, size1);

    // 16. destroy handle
    EXPECT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
}

TEST_F(checkin_misc_MEMORY_MODEL, user_managed)
{
    size_t size;
    rocblas_status status;
    rocblas_handle handle;

    /*************************************/
    /******** user fixed size  ***********/
    /*************************************/

    // set environment variable to 2MB
    ASSERT_TRUE(set_environment_variable("ROCBLAS_DEVICE_MEMORY_SIZE", "2000000"));

    // 1. create handle
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    // 2. memory size is now fixed by user via the environment variable
    EXPECT_FALSE(rocblas_is_managing_device_memory(handle));

    // 3. 2MB should be reserved
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 4. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 5. getrf small will require .5MB
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 6. stop query; required size at the end of query (.5MB)
    rocblas_stop_device_memory_size_query(handle, &size);
    EXPECT_LT(size, 2000000);

    // 7. device memory size should not change; it should be 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 8. When executing getrf. Enough memory allowing execution to success
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_success);

    // 9. device memory size should stay the same (2MB)
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 10. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 11. getrf baseline will require 54MB
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 12. stop query; required size at the end of query (54MB)
    rocblas_stop_device_memory_size_query(handle, &size);
    EXPECT_GT(size, 2000000);

    // 13. device memory size should not change; it should still be 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 14. When executing getrf, device memory is not enough for execution to success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_memory_error);

    // 15. device memory size should be the same 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 16. set mem size to 0 get rocblas the control back
    rocblas_set_device_memory_size(handle, 0);
    EXPECT_TRUE(rocblas_is_managing_device_memory(handle));

    /*************************************/
    /******** user managed size  **********/
    /*************************************/

    // 1. set mem size to 2MB
    rocblas_set_device_memory_size(handle, 2000000);
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 2. memory size should now be fixed by the user
    EXPECT_FALSE(rocblas_is_managing_device_memory(handle));

    // 3. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 4. getrf baseline will require 54MB
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 5. call getrf small, which will require less than 54MB and so size will remain unchanged
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_size_unchanged);

    // 6. stop query
    rocblas_stop_device_memory_size_query(handle, &size);
    EXPECT_GT(size, 2000000);

    // 7. device memory size should not change; it should be 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 8. When executing getrf baseline, device memory is not enough for success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_memory_error);

    // 9. device memory size should be the same 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 10. set mem size to 100MB
    rocblas_set_device_memory_size(handle, 100000000);
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 100000000);

    // 11. When executing getrf, device memory should now be enough for execution to success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_success);

    // 12. destroy handle
    EXPECT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
}

/*************************************/
/******** user owned workspace *******/
/*************************************/
TEST_F(checkin_misc_MEMORY_MODEL, user_owned)
{
    size_t size;
    rocblas_status status;
    rocblas_handle handle;

    // 1. create handle
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    // 2. by default, memory is rocblas managed
    EXPECT_TRUE(rocblas_is_managing_device_memory(handle));

    // 3. by default, 32MB should be reserved
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 32 * 1024 * 1024);

    // 4. pass user owned workspace (2MB)
    void* W;
    size_t sw = 2000000;
    hipMalloc(&W, sw);
    ASSERT_EQ(rocblas_set_workspace(handle, W, sw), rocblas_status_success);

    // 5. memory should now be user managed
    EXPECT_FALSE(rocblas_is_managing_device_memory(handle));

    // 6. 2MB should be reserved
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 7. start query
    rocblas_start_device_memory_size_query(handle);
    EXPECT_TRUE(rocblas_is_device_memory_size_query(handle));

    // 8. getrf baseline will require 54MB
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_size_increased);

    // 9. getrf small will require less than 54MB, so size should be unchanged
    status = rocsolver_dgetrf_strided_batched(handle, m_small, n_small, dA, lda, stA, dP, stP,
                                              dinfo, bc_small);
    EXPECT_EQ(status, rocblas_status_size_unchanged);

    // 10. stop query; required size at the end of query is 54MB
    rocblas_stop_device_memory_size_query(handle, &size);
    EXPECT_GT(size, 2000000);

    // 11. device memory size should not change; it should be 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 12. When executing getrf, device memory is not enough for success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_memory_error);

    // 13. device memory size should be the same 2MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 2000000);

    // 14. pass larger user owned workspace
    hipFree(W);
    sw = 100000000;
    hipMalloc(&W, sw);
    ASSERT_EQ(rocblas_set_workspace(handle, W, sw), rocblas_status_success);

    // 15. 100MB should be reserved
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 100000000);

    // 16. When executing getrf, device memory is now enough for success
    status = rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc);
    EXPECT_EQ(status, rocblas_status_success);

    // 17. device memory size should be the same 100MB
    rocblas_get_device_memory_size(handle, &size);
    EXPECT_EQ(size, 100000000);

    // 18. destroy handle
    hipFree(W);
    EXPECT_EQ(rocblas_destroy_handle(handle), rocblas_status_success);
}
