/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_labrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> labrd_tuple;

// each matrix_size_range is a {m, lda, ldx}

// each n_size_range is a {n, ldy, nb}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return (if nb = 0, else invalid)
    {0, 1, 1},
    // invalid
    {1, 1, 0},
    {-1, 1, 1},
    {1, 1, -1},
    {20, 5, 20},
    {20, 20, 5},
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 70},
    {130, 130, 150},
    {150, 200, 200}};

const vector<vector<int>> n_size_range = {
    // quick return
    {0, 1, 0},
    {1, 1, 0},
    // invalid
    {-1, 1, 1},
    {1, 1, -1},
    {20, 5, 20},
    {20, 20, 25},
    // normal (valid) samples
    {16, 16, 10},
    {20, 30, 10},
    {120, 120, 30},
    {150, 200, 30}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152, 152},
    {640, 640, 656},
    {1000, 1024, 1000},
};

const vector<vector<int>> large_n_size_range
    = {{64, 64, 60}, {98, 98, 60}, {130, 130, 100}, {220, 240, 100}, {400, 450, 100}};

Arguments labrd_setup_arguments(labrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldx", matrix_size[2]);

    arg.set<rocblas_int>("n", n_size[0]);
    arg.set<rocblas_int>("ldy", n_size[1]);
    arg.set<rocblas_int>("k", n_size[2]);

    arg.timing = 0;

    return arg;
}

class LABRD : public ::TestWithParam<labrd_tuple>
{
protected:
    LABRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = labrd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_labrd_bad_arg<T>();

        testing_labrd<T>(arg);
    }
};

// non-batch tests

TEST_P(LABRD, __float)
{
    run_tests<float>();
}

TEST_P(LABRD, __double)
{
    run_tests<double>();
}

TEST_P(LABRD, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LABRD, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LABRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LABRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
