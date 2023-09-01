/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2_getrf_npvt.hpp"
#include "testing_getrf_large.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_large_tuple;

// each matrix_size_range vector is a {N, lda, ldb}

// for weekly_lapack tests
const vector<vector<int>> very_large_matrixA_size_range = {
    {25000, 25000, 25000},
};

const vector<int> very_large_nrhs = {300};

Arguments getrf_large_setup_arguments(getrf_large_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    int nrhs = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", nrhs);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    arg.timing = 0;

    return arg;
}

class GETRF_LARGE : public ::TestWithParam<getrf_large_tuple>
{
protected:
    GETRF_LARGE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_large_setup_arguments(GetParam());

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getrf_large<BATCHED, STRIDED, true, T>(arg);
    }
};

// Make changes to this for large matrices.

class GETRF_LARGE_NPVT : public ::TestWithParam<getrf_large_tuple>
{
protected:
    GETRF_LARGE_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_large_setup_arguments(GetParam());

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getf2_getrf_npvt<BATCHED, STRIDED, true, T>(arg);
    }
};

// non-batch tests

TEST_P(GETRF_LARGE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_LARGE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_LARGE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRF_LARGE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GETRF_LARGE, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRF_LARGE, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRF_LARGE, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRF_LARGE, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GETRF_LARGE, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRF_LARGE, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRF_LARGE, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRF_LARGE, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(weekly_lapack,
                         GETRF_LARGE,
                         Combine(ValuesIn(very_large_matrixA_size_range), ValuesIn(very_large_nrhs)));
