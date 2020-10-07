/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larfb.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> larfb_tuple;

// each matrix_size vector is a {M,N,lda,s,ldv,st}
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'

// each reflector_size vector is a {K,ldt,d,t}
// if d = 0, then direct = 'F'
// if d = 1, then direct = 'B'
// if t = 0, then trans = 'N'
// if t = 1, then trans = 'T'
// if t = 2, then trans = 'C'

// case when m = 0 and k = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0, 1, 0},
    {1, 0, 1, 0, 1, 0},
    // invalid
    {-1, 1, 1, 0, 1, 0},
    {1, -1, 1, 0, 1, 0},
    {15, 15, 5, 0, 15, 0},
    {12, 5, 12, 0, 5, 0},
    {5, 12, 15, 1, 5, 0},
    {15, 10, 15, 0, 5, 1},
    // normal (valid) samples
    {15, 15, 15, 0, 15, 0},
    {18, 20, 20, 1, 20, 0},
    {20, 18, 20, 0, 20, 0},
    {20, 30, 20, 1, 30, 0},
    {50, 35, 50, 0, 50, 0},
    {40, 40, 40, 0, 15, 1},
    {40, 40, 40, 1, 25, 1}};

const vector<vector<int>> reflector_size_range = {
    // invalid
    {0, 1, 0, 0},
    {5, 1, 0, 0},
    // normal (valid) samples
    {7, 7, 0, 1},
    {10, 10, 1, 1},
    {12, 70, 0, 2},
    {15, 15, 1, 2}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 192, 0, 192, 0},   {640, 640, 640, 1, 700, 0},     {640, 640, 700, 0, 640, 0},
       {840, 1024, 840, 1, 1024, 0}, {2547, 1980, 2547, 0, 2547, 0}, {200, 200, 220, 0, 100, 1},
       {240, 300, 240, 1, 100, 1},   {600, 200, 600, 1, 100, 1}};

const vector<vector<int>> large_reflector_size_range
    = {{35, 35, 0, 1},   {50, 70, 0, 0},   {85, 85, 1, 1},
       {100, 150, 1, 0}, {100, 150, 0, 2}, {100, 150, 1, 2}};

Arguments larfb_setup_arguments(larfb_tuple tup)
{
    vector<int> order_size = std::get<0>(tup);
    vector<int> reflector_size = std::get<1>(tup);

    Arguments arg;

    arg.M = order_size[0];
    arg.N = order_size[1];
    arg.lda = order_size[2];
    arg.ldv = order_size[4];

    arg.side_option = order_size[3] == 0 ? 'L' : 'R';

    arg.K = reflector_size[0];
    arg.ldt = reflector_size[1];

    arg.direct_option = reflector_size[2] == 1 ? 'B' : 'F';
    arg.transH_option = reflector_size[3] == 0 ? 'N' : (reflector_size[3] == 1 ? 'T' : 'C');
    arg.storev = order_size[5] == 1 ? 'R' : 'C';

    arg.timing = 0;

    return arg;
}

class LARFB : public ::TestWithParam<larfb_tuple>
{
protected:
    LARFB() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(LARFB, __float)
{
    Arguments arg = larfb_setup_arguments(GetParam());

    if(arg.M == 0 && arg.K == 0)
        testing_larfb_bad_arg<float>();

    testing_larfb<float>(arg);
}

TEST_P(LARFB, __double)
{
    Arguments arg = larfb_setup_arguments(GetParam());

    if(arg.M == 0 && arg.K == 0)
        testing_larfb_bad_arg<double>();

    testing_larfb<double>(arg);
}

TEST_P(LARFB, __float_complex)
{
    Arguments arg = larfb_setup_arguments(GetParam());

    if(arg.M == 0 && arg.K == 0)
        testing_larfb_bad_arg<rocblas_float_complex>();

    testing_larfb<rocblas_float_complex>(arg);
}

TEST_P(LARFB, __double_complex)
{
    Arguments arg = larfb_setup_arguments(GetParam());

    if(arg.M == 0 && arg.K == 0)
        testing_larfb_bad_arg<rocblas_double_complex>();

    testing_larfb<rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LARFB,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(large_reflector_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARFB,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(reflector_size_range)));
