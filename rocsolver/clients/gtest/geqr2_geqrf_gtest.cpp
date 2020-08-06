/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_geqr2_geqrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef std::tuple<vector<int>, int> geqr_tuple;

// each matrix_size_range is a {m, lda}

// case when m = n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    {0, 1},             //quick return 
    {-1, 1}, {20, 5},   //invalid
    {50, 50}, {70, 100}, {130, 130}, {150, 200}
};

const vector<int> n_size_range = {
    0,  //quick return
    -1, //invalid
    16, 20, 130, 150
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152}, {640, 640}, {1000, 1024}, 
};

const vector<int> large_n_size_range = {
    64, 98, 130, 220, 400
};


Arguments setup_arguments_qr(geqr_tuple tup) 
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

class GEQR2 : public ::TestWithParam<geqr_tuple> {
protected:
    GEQR2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GEQRF : public ::TestWithParam<geqr_tuple> {
protected:
    GEQRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};


// non-batch tests

TEST_P(GEQR2, __float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,0,float>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,0,float>(arg);
}

TEST_P(GEQR2, __double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,0,double>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,0,double>(arg);
}

TEST_P(GEQR2, __float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,0,rocblas_float_complex>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,0,rocblas_float_complex>(arg);
}

TEST_P(GEQR2, __double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,0,rocblas_double_complex>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,0,rocblas_double_complex>(arg);
}

TEST_P(GEQRF, __float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,1,float>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,1,float>(arg);
}

TEST_P(GEQRF, __double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,1,double>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,1,double>(arg);
}

TEST_P(GEQRF, __float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,1,rocblas_float_complex>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,1,rocblas_float_complex>(arg);
}

TEST_P(GEQRF, __double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,false,1,rocblas_double_complex>();

    arg.batch_count = 1;
    testing_geqr2_geqrf<false,false,1,rocblas_double_complex>(arg);
}


// batched tests

TEST_P(GEQR2, batched__float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,0,float>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,0,float>(arg);
}

TEST_P(GEQR2, batched__double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,0,double>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,0,double>(arg);
}

TEST_P(GEQR2, batched__float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,0,rocblas_float_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,0,rocblas_float_complex>(arg);
}

TEST_P(GEQR2, batched__double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,0,rocblas_double_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,0,rocblas_double_complex>(arg);
}

TEST_P(GEQRF, batched__float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,1,float>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,1,float>(arg);
}

TEST_P(GEQRF, batched__double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,1,double>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,1,double>(arg);
}

TEST_P(GEQRF, batched__float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,1,rocblas_float_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,1,rocblas_float_complex>(arg);
}

TEST_P(GEQRF, batched__double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,true,1,rocblas_double_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,true,1,rocblas_double_complex>(arg);
}


// strided_batched cases

TEST_P(GEQR2, strided_batched__float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,0,float>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,0,float>(arg);
}

TEST_P(GEQR2, strided_batched__double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,0,double>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,0,double>(arg);
}

TEST_P(GEQR2, strided_batched__float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,0,rocblas_float_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,0,rocblas_float_complex>(arg);
}

TEST_P(GEQR2, strided_batched__double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,0,rocblas_double_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,0,rocblas_double_complex>(arg);
}

TEST_P(GEQRF, strided_batched__float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,1,float>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,1,float>(arg);
}

TEST_P(GEQRF, strided_batched__double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,1,double>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,1,double>(arg);
}

TEST_P(GEQRF, strided_batched__float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,1,rocblas_float_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,1,rocblas_float_complex>(arg);
}

TEST_P(GEQRF, strided_batched__double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<false,true,1,rocblas_double_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<false,true,1,rocblas_double_complex>(arg);
}


// ptr_batched tests

TEST_P(GEQRF, ptr_batched__float) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,false,1,float>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,false,1,float>(arg);
}

TEST_P(GEQRF, ptr_batched__double) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,false,1,double>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,false,1,double>(arg);
}

TEST_P(GEQRF, ptr_batched__float_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,false,1,rocblas_float_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,false,1,rocblas_float_complex>(arg);
}

TEST_P(GEQRF, ptr_batched__double_complex) {
    Arguments arg = setup_arguments_qr(GetParam());

    if (arg.M == 0 && arg.N == 0)
        testing_geqr2_geqrf_bad_arg<true,false,1,rocblas_double_complex>();

    arg.batch_count = 3;
    testing_geqr2_geqrf<true,false,1,rocblas_double_complex>(arg);
}


INSTANTIATE_TEST_SUITE_P(daily_lapack, GEQR2,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEQR2,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEQRF,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEQRF,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(n_size_range)));


