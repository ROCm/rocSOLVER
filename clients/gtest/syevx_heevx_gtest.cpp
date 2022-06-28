/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syevx_heevx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> syevx_heevx_tuple;

// each size_range vector is a {n, lda, ldz, vl, vu, il, iu}

// each op_range vector is a {evect, erange, uplo}

// case when n == 0, evect == N, erange == V and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> op_range
    = {{'N', 'V', 'L'}, {'V', 'A', 'U'}, {'V', 'V', 'L'}, {'V', 'I', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 1, 1, 0, 10, 1, 0},
    // invalid
    {-1, 1, 1, 0, 10, 1, 1},
    {10, 5, 10, 0, 10, 1, 1},
    {10, 10, 5, 0, 10, 1, 1},
    // valid only when erange=A
    {10, 10, 10, 10, 0, 10, 1},
    // normal (valid) samples
    {1, 1, 1, 0, 10, 1, 1},
    {12, 12, 15, -20, 20, 10, 12},
    {20, 30, 30, 5, 15, 1, 20},
    {35, 35, 35, -10, 10, 1, 15},
    {50, 60, 50, -15, -5, 20, 30}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{192, 192, 192, 5, 15, 100, 170},
                                              {256, 270, 256, -10, 10, 1, 256},
                                              {300, 300, 330, -15, -5, 200, 300}};

template <typename T>
Arguments syevx_heevx_setup_arguments(syevx_heevx_tuple tup)
{
    using S = decltype(std::real(T{}));

    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    arg.set<rocblas_int>("ldz", size[2]);
    arg.set<double>("vl", size[3]);
    arg.set<double>("vu", size[4]);
    arg.set<rocblas_int>("il", size[5]);
    arg.set<rocblas_int>("iu", size[6]);

    arg.set<char>("evect", op[0]);
    arg.set<char>("erange", op[1]);
    arg.set<char>("uplo", op[2]);

    arg.set<double>("abstol", 0);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class SYEVX_HEEVX : public ::TestWithParam<syevx_heevx_tuple>
{
protected:
    SYEVX_HEEVX() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        using S = decltype(std::real(T{}));

        Arguments arg = syevx_heevx_setup_arguments<T>(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("erange") == 'V' && arg.peek<char>("uplo") == 'L')
            testing_syevx_heevx_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevx_heevx<BATCHED, STRIDED, T>(arg);
    }
};

class SYEVX : public SYEVX_HEEVX
{
};

class HEEVX : public SYEVX_HEEVX
{
};

// non-batch tests

TEST_P(SYEVX, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVX, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVX, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVX, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYEVX, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYEVX, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEEVX, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEEVX, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEVX, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVX, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVX, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVX, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, SYEVX, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVX, Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, HEEVX, Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVX, Combine(ValuesIn(size_range), ValuesIn(op_range)));
