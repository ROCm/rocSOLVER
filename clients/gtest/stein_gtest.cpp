/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_stein.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> stein_tuple;

// each size_range vector is a {N, ldz}

// each vec_range is a {nev}
// Indicates the number of vectors to compute
// (vectors are always associated with the last nev eigenvalues)

// case when N == 0 and nev == 5 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {2, 1},
    // normal (valid) samples
    {15, 15},
    {20, 30},
    {35, 40}};
const vector<int> vec_range = {5, 10, 15};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{192, 192}, {256, 270}, {300, 300}};
const vector<int> large_vec_range = {25, 40, 65};

Arguments stein_setup_arguments(stein_tuple tup)
{
    Arguments arg;

    vector<int> size = std::get<0>(tup);
    rocblas_int nev = std::get<1>(tup);

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldz", size[1]);
    arg.set<rocblas_int>("nev", nev);

    arg.timing = 0;

    return arg;
}

class STEIN : public ::TestWithParam<stein_tuple>
{
protected:
    STEIN() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = stein_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nev") == 5)
            testing_stein_bad_arg<T>();

        testing_stein<T>(arg);
    }
};

// non-batch tests

TEST_P(STEIN, __float)
{
    run_tests<float>();
}

TEST_P(STEIN, __double)
{
    run_tests<double>();
}

TEST_P(STEIN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(STEIN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         STEIN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_vec_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEIN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(vec_range)));
