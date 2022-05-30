/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syevdx_heevdx_inplace.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> syevdx_heevdx_tuple;

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
Arguments syevdx_heevdx_setup_arguments(syevdx_heevdx_tuple tup, bool inplace)
{
    using S = decltype(std::real(T{}));

    vector<int> size = std::get<0>(tup);
    vector<printable_char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    if(!inplace)
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

class SYEVDX_HEEVDX_INPLACE : public ::TestWithParam<syevdx_heevdx_tuple>
{
protected:
    SYEVDX_HEEVDX_INPLACE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        using S = decltype(std::real(T{}));

        Arguments arg = syevdx_heevdx_setup_arguments<T>(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("erange") == 'V' && arg.peek<char>("uplo") == 'L')
            testing_syevdx_heevdx_inplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevdx_heevdx_inplace<BATCHED, STRIDED, T>(arg);
    }
};

class SYEVDX_INPLACE : public SYEVDX_HEEVDX_INPLACE
{
};

class HEEVDX_INPLACE : public SYEVDX_HEEVDX_INPLACE
{
};

// non-batch tests

TEST_P(SYEVDX_INPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVDX_INPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVDX_INPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVDX_INPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYEVDX_INPLACE,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVDX_INPLACE,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEEVDX_INPLACE,
                         Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVDX_INPLACE,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
