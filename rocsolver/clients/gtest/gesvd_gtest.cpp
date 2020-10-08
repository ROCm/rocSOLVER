/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesvd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvd_tuple;

// each size_range vector is a {m, n, fa};
// if fa = 0 then no fast algorithm is allowed
// if fa = 1 fast algorithm is used when possible

// each opt_range vector is a {lda, ldu, ldv, leftsv, rightsv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit
// if leftsv (rightsv) = 0 then overwrite singular vectors
// if leftsv (rightsv) = 1 then compute singular vectors
// if leftsv (rightsv) = 2 then compute all orthogonal matrix
// if leftsv (rightsv) = 3 then no singular vectors are computed

// case when m = n = 0 and rightsv = leftsv = 3 will also execute the bad
// arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // quick return
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    // invalid
    {-1, 1, 0},
    {1, -1, 0},
    // normal (valid) samples
    {20, 20, 0},
    {40, 30, 0},
    {60, 30, 0},
    {60, 30, 1},
    {30, 40, 0},
    {30, 60, 0},
    {30, 60, 1}};

const vector<vector<int>> opt_range = {
    // invalid
    {-1, 0, 0, 2, 2},
    {0, -1, 0, 1, 2},
    {0, 0, -1, 2, 1},
    {0, -1, 0, 2, 2},
    {0, 0, -1, 2, 2},
    // normal (valid) samples
    {1, 1, 1, 3, 3},
    {0, 0, 0, 2, 2},
    {1, 0, 0, 0, 1},
    {0, 1, 0, 1, 0},
    {0, 0, 1, 1, 1},
    {0, 0, 0, 3, 0},
    {0, 0, 0, 3, 1},
    {0, 0, 0, 3, 2},
    {0, 0, 0, 0, 3},
    {0, 0, 0, 1, 3},
    {0, 0, 0, 2, 3}};

// for daily_lapack tests
const vector<vector<int>> large_size_range
    = {{120, 100, 0}, {300, 120, 0}, {300, 120, 1}, {100, 120, 0}, {120, 300, 0}, {120, 300, 1}};

const vector<vector<int>> large_opt_range
    = {{0, 0, 0, 3, 3}, {1, 0, 0, 0, 1}, {0, 1, 0, 1, 0}, {0, 0, 1, 1, 1},
       {0, 0, 0, 3, 0}, {0, 0, 0, 1, 3}, {0, 0, 0, 3, 2}};

Arguments gesvd_setup_arguments(gesvd_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt = std::get<1>(tup);

    Arguments arg;

    // sizes
    arg.M = size[0];
    arg.N = size[1];

    // fast algorithm
    if(size[2] == 0)
        arg.workmode = 'I';
    else
        arg.workmode = 'O';

    // leading dimensions
    arg.lda = arg.M; // lda
    arg.ldb = arg.M; // ldu
    arg.ldv = opt[4] == 2 ? arg.N : min(arg.M, arg.N); // ldv
    arg.lda += opt[0] * 10;
    arg.ldb += opt[1] * 10;
    arg.ldv += opt[2] * 10;

    // vector options
    if(opt[3] == 0)
        arg.left_svect = 'O';
    else if(opt[3] == 1)
        arg.left_svect = 'S';
    else if(opt[3] == 2)
        arg.left_svect = 'A';
    else
        arg.left_svect = 'N';

    if(opt[4] == 0)
        arg.right_svect = 'O';
    else if(opt[4] == 1)
        arg.right_svect = 'S';
    else if(opt[4] == 2)
        arg.right_svect = 'A';
    else
        arg.right_svect = 'N';

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N; // strideA
    arg.bsb = min(arg.M, arg.N); // strideS
    arg.bsc = arg.ldb * arg.M; // strideU
    arg.bsp = arg.ldv * arg.N; // strideV
    arg.bs5 = arg.bsb; // strideE

    arg.timing = 0;

    return arg;
}

class GESVD : public ::TestWithParam<gesvd_tuple>
{
protected:
    GESVD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(GESVD, __float)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, false, float>();

    arg.batch_count = 1;
    testing_gesvd<false, false, float>(arg);
}

TEST_P(GESVD, __double)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, false, double>();

    arg.batch_count = 1;
    testing_gesvd<false, false, double>(arg);
}

TEST_P(GESVD, __float_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_gesvd<false, false, rocblas_float_complex>(arg);
}

TEST_P(GESVD, __double_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_gesvd<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(GESVD, batched__float)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<true, true, float>();

    arg.batch_count = 3;
    testing_gesvd<true, true, float>(arg);
}

TEST_P(GESVD, batched__double)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<true, true, double>();

    arg.batch_count = 3;
    testing_gesvd<true, true, double>(arg);
}

TEST_P(GESVD, batched__float_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gesvd<true, true, rocblas_float_complex>(arg);
}

TEST_P(GESVD, batched__double_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gesvd<true, true, rocblas_double_complex>(arg);
}

// strided_batched tests

TEST_P(GESVD, strided_batched__float)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, true, float>();

    arg.batch_count = 3;
    testing_gesvd<false, true, float>(arg);
}

TEST_P(GESVD, strided_batched__double)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, true, double>();

    arg.batch_count = 3;
    testing_gesvd<false, true, double>(arg);
}

TEST_P(GESVD, strided_batched__float_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_gesvd<false, true, rocblas_float_complex>(arg);
}

TEST_P(GESVD, strided_batched__double_complex)
{
    Arguments arg = gesvd_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0 && arg.left_svect == 'N' && arg.right_svect == 'N')
        testing_gesvd_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_gesvd<false, true, rocblas_double_complex>(arg);
}

// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESVD,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_SUITE_P(checkin_lapack, GESVD, Combine(ValuesIn(size_range), ValuesIn(opt_range)));
