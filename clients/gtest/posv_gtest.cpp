/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_posv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> posv_tuple;

// each A_range vector is a {N, lda, ldb, singular};
// if singular = 1, then the used matrix for the tests is not positive definite

// each B_range vector is a {nrhs, uplo};
// if uplo = 0 then upper
// if uplo = 1 then lower

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
const vector<vector<int>> matrix_sizeB_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 1},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range = {{70, 70, 100, 0},
                                                      {192, 192, 192, 1},
                                                      {600, 700, 645, 0},
                                                      {1000, 1000, 1000, 1},
                                                      {1000, 2000, 2000, 0}};
const vector<vector<int>> large_matrix_sizeB_range = {
    {100, 0}, {150, 0}, {200, 1}, {524, 1}, {1000, 0},
};

Arguments posv_setup_arguments(posv_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB[0]);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    if(matrix_sizeB[1] == 0)
        arg.set<char>("uplo", 'U');
    else
        arg.set<char>("uplo", 'L');

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_sizeA[3];

    return arg;
}

class POSV : public ::TestWithParam<posv_tuple>
{
protected:
    POSV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = posv_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nrhs") == 0)
            testing_posv_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_posv<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_posv<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(POSV, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POSV, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POSV, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POSV, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POSV, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(POSV, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(POSV, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(POSV, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(POSV, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(POSV, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(POSV, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(POSV, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POSV,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POSV,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
