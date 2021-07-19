/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesv.hpp"
#include "testing_gesv_outofplace.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gesv_tuple;

// each A_range vector is a {N, lda, ldb/ldx, singular};
// if singular = 1, then the used matrix for the tests is singular

// each B_range vector is a {nrhs};

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {10, 2, 10, 0},
    {10, 10, 2, 0},
    /// normal (valid) samples
    {20, 20, 20, 0},
    {30, 50, 30, 1},
    {30, 30, 50, 0},
    {50, 60, 60, 1}};
const vector<int> matrix_sizeB_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    10,
    20,
    30,
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range = {{70, 70, 100, 0},
                                                      {192, 192, 192, 1},
                                                      {600, 700, 645, 0},
                                                      {1000, 1000, 1000, 1},
                                                      {1000, 2000, 2000, 0}};
const vector<int> large_matrix_sizeB_range = {
    100, 150, 200, 524, 1000,
};

Arguments gesv_setup_arguments(gesv_tuple tup, bool outofplace)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    int matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    if(outofplace)
        arg.set<rocblas_int>("ldx", matrix_sizeA[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_sizeA[3];

    return arg;
}

class GESV : public ::TestWithParam<gesv_tuple>
{
protected:
    GESV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesv_setup_arguments(GetParam(), false);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nrhs") == 0)
            testing_gesv_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_gesv<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_gesv<BATCHED, STRIDED, T>(arg);
    }
};

class GESV_OUTOFPLACE : public ::TestWithParam<gesv_tuple>
{
protected:
    GESV_OUTOFPLACE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesv_setup_arguments(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nrhs") == 0)
            testing_gesv_outofplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_gesv_outofplace<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_gesv_outofplace<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GESV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESV_OUTOFPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESV_OUTOFPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESV_OUTOFPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESV_OUTOFPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GESV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GESV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GESV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GESV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GESV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GESV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GESV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GESV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESV,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESV_OUTOFPLACE,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESV_OUTOFPLACE,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
