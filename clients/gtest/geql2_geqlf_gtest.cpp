/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_geql2_geqlf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> geqlf_tuple;

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

Arguments geqlf_setup_arguments(geqlf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = n_size;
    arg.lda = matrix_size[1];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = min(arg.M, arg.N);
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GEQL2_GEQLF : public ::TestWithParam<geqlf_tuple>
{
protected:
    GEQL2_GEQLF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, bool BLOCKED, typename T>
    void test_fixture()
    {
        Arguments arg = geqlf_setup_arguments(GetParam());

        if(arg.M == 0 && arg.N == 0)
            testing_geql2_geqlf_bad_arg<BATCHED, STRIDED, BLOCKED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_geql2_geqlf<BATCHED, STRIDED, BLOCKED, T>(arg);
    }
};

class GEQL2 : public GEQL2_GEQLF
{
};

class GEQLF : public GEQL2_GEQLF
{
};

// non-batch tests

TEST_P(GEQL2, __float)
{
    test_fixture<false, false, 0, float>();
}

TEST_P(GEQL2, __double)
{
    test_fixture<false, false, 0, double>();
}

TEST_P(GEQL2, __float_complex)
{
    test_fixture<false, false, 0, rocblas_float_complex>();
}

TEST_P(GEQL2, __double_complex)
{
    test_fixture<false, false, 0, rocblas_double_complex>();
}

TEST_P(GEQLF, __float)
{
    test_fixture<false, false, 1, float>();
}

TEST_P(GEQLF, __double)
{
    test_fixture<false, false, 1, double>();
}

TEST_P(GEQLF, __float_complex)
{
    test_fixture<false, false, 1, rocblas_float_complex>();
}

TEST_P(GEQLF, __double_complex)
{
    test_fixture<false, false, 1, rocblas_double_complex>();
}

// batched tests

TEST_P(GEQL2, batched__float)
{
    test_fixture<true, true, 0, float>();
}

TEST_P(GEQL2, batched__double)
{
    test_fixture<true, true, 0, double>();
}

TEST_P(GEQL2, batched__float_complex)
{
    test_fixture<true, true, 0, rocblas_float_complex>();
}

TEST_P(GEQL2, batched__double_complex)
{
    test_fixture<true, true, 0, rocblas_double_complex>();
}

TEST_P(GEQLF, batched__float)
{
    test_fixture<true, true, 1, float>();
}

TEST_P(GEQLF, batched__double)
{
    test_fixture<true, true, 1, double>();
}

TEST_P(GEQLF, batched__float_complex)
{
    test_fixture<true, true, 1, rocblas_float_complex>();
}

TEST_P(GEQLF, batched__double_complex)
{
    test_fixture<true, true, 1, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(GEQL2, strided_batched__float)
{
    test_fixture<false, true, 0, float>();
}

TEST_P(GEQL2, strided_batched__double)
{
    test_fixture<false, true, 0, double>();
}

TEST_P(GEQL2, strided_batched__float_complex)
{
    test_fixture<false, true, 0, rocblas_float_complex>();
}

TEST_P(GEQL2, strided_batched__double_complex)
{
    test_fixture<false, true, 0, rocblas_double_complex>();
}

TEST_P(GEQLF, strided_batched__float)
{
    test_fixture<false, true, 1, float>();
}

TEST_P(GEQLF, strided_batched__double)
{
    test_fixture<false, true, 1, double>();
}

TEST_P(GEQLF, strided_batched__float_complex)
{
    test_fixture<false, true, 1, rocblas_float_complex>();
}

TEST_P(GEQLF, strided_batched__double_complex)
{
    test_fixture<false, true, 1, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQL2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQL2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEQLF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEQLF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
