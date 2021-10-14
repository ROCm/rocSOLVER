/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_lasyf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> lasyf_tuple;

// each matrix_size_range is a {n, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// each op_range is a {nb, ul}
// if ul = 0, then uplo = 'L'
// if ul = 1, then uplo = 'U'

// case when n = 0 and nb = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    {20, 20, 0},
    // normal (valid) samples
    {35, 50, 0},
    {70, 100, 1},
    {130, 130, 0},
    {150, 150, 1}};

const vector<vector<int>> op_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    {180, 0},
    // normal (valid) samples
    {10, 0},
    {25, 1},
    {30, 0},
    {30, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{152, 152, 1}, {640, 640, 0}, {1000, 1024, 1}};

const vector<vector<int>> large_op_range = {{64, 0}, {98, 1}, {130, 0}, {150, 1}};

Arguments lasyf_setup_arguments(lasyf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> op_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<rocblas_int>("nb", op_size[0]);
    arg.set<char>("uplo", op_size[1] ? 'U' : 'L');

    arg.timing = 0;
    arg.singular = matrix_size[2];

    return arg;
}

class LASYF : public ::TestWithParam<lasyf_tuple>
{
protected:
    LASYF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = lasyf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_lasyf_bad_arg<T>();

        if(arg.singular == 1)
            testing_lasyf<T>(arg);

        arg.singular = 0;
        testing_lasyf<T>(arg);
    }
};

// non-batch tests

TEST_P(LASYF, __float)
{
    run_tests<float>();
}

TEST_P(LASYF, __double)
{
    run_tests<double>();
}

TEST_P(LASYF, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LASYF, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LASYF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LASYF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));
