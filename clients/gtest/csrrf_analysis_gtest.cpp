/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrrf_analysis.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int> csrrf_analysis_tuple;

// case when n = 0 and nnz = 60 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<int> n_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    20,
    50,
};
const vector<int> nnz_range = {
    // invalid
    -1,
    // normal (valid) samples
    60,
    100,
    140,
};

// for daily_lapack tests
const vector<int> large_n_range = {
    // normal (valid) samples
    100,
    250,
};
const vector<int> large_nnz_range = {
    // normal (valid) samples
    300,
    500,
    700,
};

Arguments csrrf_analysis_setup_arguments(csrrf_analysis_tuple tup)
{
    int n = std::get<0>(tup);
    int nnz = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n);
    arg.set<rocblas_int>("nnzM", nnz);
    arg.set<rocblas_int>("nnzT", nnz);
    // note: the clients will determine the test case with n and nnzM.
    // nnzT = nnz is passed because it does not have a default value in the
    // bench client (for future purposes).

    arg.timing = 0;

    return arg;
}

class CSRRF_ANALYSIS : public ::TestWithParam<csrrf_analysis_tuple>
{
protected:
    CSRRF_ANALYSIS() {}
    virtual void SetUp()
    {
        if(rocsolver_create_rfinfo(nullptr, nullptr) == rocblas_status_not_implemented)
            GTEST_SKIP() << "Sparse functionality is not enabled";
    }
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrrf_analysis_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<rocblas_int>("nnzM") == 60)
            testing_csrrf_analysis_bad_arg<T>();

        testing_csrrf_analysis<T>(arg);
    }
};

// non-batch tests

TEST_P(CSRRF_ANALYSIS, __float)
{
    run_tests<float>();
}

TEST_P(CSRRF_ANALYSIS, __double)
{
    run_tests<double>();
}

/*TEST_P(CSRRF_ANALYSIS, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(CSRRF_ANALYSIS, __double_complex)
{
    run_tests<rocblas_double_complex>();
}*/

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRRF_ANALYSIS,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRRF_ANALYSIS,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
