/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrrf_workflow.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>, vector<printable_char>> csrrf_workflow_tuple;

// each n_range vector is {n, ldb}

// each nnz_range vector is {nnzT, nrhs}

// each mode_range is a {mode, analysis_mode}
// if mode = '1', then the factorization is LU
// if mode = '2', then the factorization is Cholesky

// for checkin_lapack tests
const vector<vector<int>> n_range = {
    // normal (valid) samples
    {20, 20},
};
const vector<vector<int>> nnz_range = {
    // normal (valid) samples
    {60, 10},
};

const vector<vector<printable_char>> mode_range = {
    {'1', '2'},
    {'2', '1'},
};

Arguments csrrf_workflow_setup_arguments(csrrf_workflow_tuple tup)
{
    vector<int> n_v = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);
    vector<printable_char> mode_v = std::get<2>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v[0]);
    arg.set<rocblas_int>("ldb", n_v[1]);
    arg.set<rocblas_int>("nnzM", nnz_v[0]);
    arg.set<rocblas_int>("nnzT", nnz_v[0]);
    arg.set<rocblas_int>("nrhs", nnz_v[1]);
    arg.set<char>("rfinfo_mode", mode_v[0]);
    arg.set<char>("analysis_mode", mode_v[1]);
    // note: the clients will determine the test case with n and nnzM.
    // nnzT = nnz is passed because it does not have a default value in the
    // bench client (for future purposes).

    arg.timing = 0;

    return arg;
}

class CSRRF_WORKFLOW : public ::TestWithParam<csrrf_workflow_tuple>
{
protected:
    CSRRF_WORKFLOW() {}
    virtual void SetUp()
    {
        if(rocsolver_create_rfinfo(nullptr, nullptr) == rocblas_status_not_implemented)
            GTEST_SKIP() << "Sparse functionality is not enabled";
    }
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_workflow_setup_arguments(GetParam());

        testing_csrrf_workflow<T>(arg);
    }
};

// non-batch tests

TEST_P(CSRRF_WORKFLOW, __float)
{
    run_tests<float>();
}

TEST_P(CSRRF_WORKFLOW, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_WORKFLOW, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_WORKFLOW, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRRF_WORKFLOW,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range), ValuesIn(mode_range)));
