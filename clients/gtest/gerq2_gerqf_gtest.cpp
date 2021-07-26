/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gerq2_gerqf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gerqf_tuple;

// each matrix_size_range is a {m, lda}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16, 20, 130, 150};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {64, 98, 130, 220, 400};

Arguments gerqf_setup_arguments(gerqf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool BLOCKED>
class GERQ2_GERQF : public ::TestWithParam<gerqf_tuple>
{
protected:
    GERQ2_GERQF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gerqf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_gerq2_gerqf_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gerq2_gerqf<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class GERQ2 : public GERQ2_GERQF<false>
{
};

class GERQF : public GERQ2_GERQF<true>
{
};

// non-batch tests

TEST_P(GERQ2, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GERQ2, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GERQ2, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GERQ2, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GERQF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GERQF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GERQF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GERQF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GERQ2, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GERQ2, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GERQ2, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GERQ2, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GERQF, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GERQF, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GERQF, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GERQF, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GERQ2, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GERQ2, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GERQ2, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GERQ2, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GERQF, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GERQF, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GERQF, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GERQF, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GERQ2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GERQ2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GERQF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GERQF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
