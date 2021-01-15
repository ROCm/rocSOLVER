/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gels.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int, int, int, int> gels_params_A;
typedef std::tuple<int, rocsolver_op_char> gels_params_B;

typedef std::tuple<gels_params_A, gels_params_B> gels_tuple;

// each A_range tuple is a {M, N, lda, ldb, rankd};

// each B_range tuple is a {nrhs, trans};

// case when N = nrhs = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<gels_params_A> matrix_sizeA_range = {
    // quick return
    {0, 0, 0, 0, 0},
    // invalid
    {-1, 1, 1, 1, 0},
    {1, -1, 1, 1, 0},
    {10, 10, 10, 1, 0},
    {10, 10, 1, 10, 0},
    // not yet implemented
    {10, 1, 10, 10, 0},
    // normal (valid) samples
    {20, 20, 20, 20, 0},
    {20, 20, 20, 20, 1},
    {30, 20, 40, 30, 0},
    {30, 20, 40, 30, 1},
    {40, 20, 40, 40, 0},
    {40, 20, 40, 40, 1},
};
const vector<gels_params_B> matrix_sizeB_range = {
    // quick return
    {0, 'N'},
    // invalid
    {-1, 'N'},
    // not yet implemented
    {1, 'T'},
    {1, 'C'},
    // normal (valid) samples
    {10, 'N'},
    {20, 'N'},
    {30, 'N'},
};

// for daily_lapack tests
const vector<gels_params_A> large_matrix_sizeA_range = {
    {75, 25, 75, 75, 0},    {150, 150, 150, 150, 0},    {150, 150, 150, 150, 1},
    {500, 50, 600, 600, 0}, {1000, 500, 1000, 1000, 0},
};
const vector<gels_params_B> large_matrix_sizeB_range = {
    {100, 'N'},
    {200, 'N'},
    {500, 'N'},
    {1000, 'N'},
};

Arguments gels_setup_arguments(gels_tuple tup)
{
    gels_params_A matrix_sizeA = std::get<0>(tup);
    gels_params_B matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.M = std::get<0>(matrix_sizeA);
    arg.N = std::get<1>(matrix_sizeA);
    arg.lda = std::get<2>(matrix_sizeA);
    arg.ldb = std::get<3>(matrix_sizeA);
    arg.rankd = std::get<4>(matrix_sizeA);

    arg.K = std::get<0>(matrix_sizeB);
    arg.transA_option = std::get<1>(matrix_sizeB);

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;
    arg.bsb = arg.ldb * arg.K;

    return arg;
}

class GELS : public ::TestWithParam<gels_tuple>
{
protected:
    GELS() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(GELS, __float)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, false, float>();

    arg.batch_count = 1;
    testing_gels<false, false, float>(arg);
}

TEST_P(GELS, __double)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, false, double>();

    arg.batch_count = 1;
    testing_gels<false, false, double>(arg);
}

TEST_P(GELS, __float_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_gels<false, false, rocblas_float_complex>(arg);
}

TEST_P(GELS, __double_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_gels<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(GELS, batched__float)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<true, true, float>();

    arg.batch_count = 3;
    testing_gels<true, true, float>(arg);
}

TEST_P(GELS, batched__double)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<true, true, double>();

    arg.batch_count = 3;
    testing_gels<true, true, double>(arg);
}

TEST_P(GELS, batched__float_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gels<true, true, rocblas_float_complex>(arg);
}

TEST_P(GELS, batched__double_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gels<true, true, rocblas_double_complex>(arg);
}

// strided_batched tests

TEST_P(GELS, strided_batched__float)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, true, float>();

    arg.batch_count = 3;
    testing_gels<false, true, float>(arg);
}

TEST_P(GELS, strided_batched__double)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, true, double>();

    arg.batch_count = 3;
    testing_gels<false, true, double>(arg);
}

TEST_P(GELS, strided_batched__float_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gels<false, true, rocblas_float_complex>(arg);
}

TEST_P(GELS, strided_batched__double_complex)
{
    Arguments arg = gels_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_gels_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gels<false, true, rocblas_double_complex>(arg);
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GELS,
                         Combine(ValuesIn(large_matrix_sizeA_range),
                                 ValuesIn(large_matrix_sizeB_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
