/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syevj_heevj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> syevj_heevj_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {evect, uplo}

// case when n == 0, evect == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {10, 5},
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192}, {300, 310}, {515, 515}};

Arguments syevj_heevj_setup_arguments(syevj_heevj_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("evect", op[0]);
    arg.set<char>("uplo", op[1]);

    // only need to test the sorted case
    arg.set<char>("esort", 'A');

    arg.set<double>("abstol", 0);
    arg.set<rocblas_int>("max_sweeps", 100);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class SYEVJ_HEEVJ : public ::TestWithParam<syevj_heevj_tuple>
{
protected:
    SYEVJ_HEEVJ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syevj_heevj_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syevj_heevj_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevj_heevj<BATCHED, STRIDED, T>(arg);
    }
};

class SYEVJ : public SYEVJ_HEEVJ
{
};

class HEEVJ : public SYEVJ_HEEVJ
{
};

// non-batch tests

TEST_P(SYEVJ, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVJ, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVJ, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVJ, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYEVJ, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYEVJ, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEEVJ, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEEVJ, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEVJ, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVJ, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVJ, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVJ, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack, SYEVJ, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, HEEVJ, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVJ, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVJ, Combine(ValuesIn(size_range), ValuesIn(op_range)));
