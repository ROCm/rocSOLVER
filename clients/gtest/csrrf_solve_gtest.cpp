/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrrf_solve.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> csrrf_solve_tuple;

// each n_range vector is {n, ldb}

// each nnz_range vector is {nnzT, nrhs}

// case when n = 0 and nnz = 0 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> n_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 2},
    // normal (valid) samples
    {20, 30},
    {50, 60},
    {100, 100},
    {300, 300},
};
const vector<vector<int>> nnz_range = {
    // quick return
    {10, 0},
    // matrix zero
    //    {0, 1},
    // invalid
    {-1, 1},
    {10, -1},
    // normal (valid) samples
    {20, 1},
    {20, 10},
    {20, 30},
    {40, 1},
    {40, 10},
    {40, 30},
    {75, 1},
    {75, 10},
    {75, 30},
};

// for daily_lapack tests
const vector<vector<int>> large_n_range = {
    // normal (valid) samples
    {20, 30},
    {50, 50},
    {100, 110},
    {300, 300},
};
const vector<vector<int>> large_nnz_range = {
    // normal (valid) samples
    {150, 1}, {150, 10}, {150, 30}, {250, 1}, {250, 10}, {250, 30},
};

Arguments csrrf_solve_setup_arguments(csrrf_solve_tuple tup)
{
    vector<int> n_v = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v[0]);
    arg.set<rocblas_int>("ldb", n_v[1]);
    arg.set<rocblas_int>("nnzT", nnz_v[0]);
    arg.set<rocblas_int>("nrhs", nnz_v[1]);

    arg.timing = 0;

    return arg;
}

class CSRRF_SOLVE : public ::TestWithParam<csrrf_solve_tuple>
{
protected:
    CSRRF_SOLVE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_solve_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nnzT") == 0)
            testing_csrrf_solve_bad_arg<T>();

        testing_csrrf_solve<T>(arg);
    }
};

// non-batch tests

TEST_P(CSRRF_SOLVE, __float)
{
    run_tests<float>();
}

TEST_P(CSRRF_SOLVE, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_SOLVE, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_SOLVE, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRRF_SOLVE,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, CSRRF_SOLVE, Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
