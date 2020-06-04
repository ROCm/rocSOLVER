/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef vector<int> getri_tuple;

// each matrix_size_range vector is a {n, lda}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    {0, 1},             //quick return
    {-1, 1}, {20, 5},   //invalid
    {32, 32}, {50, 50}, {70, 100}, {100, 150}
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {640, 640}, {1000, 1024}, 
};


Arguments setup_arguments(getri_tuple tup) {
    //vector<int> matrix_size = std::get<0>(tup);

    Arguments arg;

    arg.N = tup[0];
    arg.lda = tup[1];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = arg.N;
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GETRI : public ::TestWithParam<getri_tuple> {
protected:
    GETRI() {}
    virtual ~GETRI() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};


// non-batch tests

TEST_P(GETRI, __float) {
    Arguments arg = setup_arguments(GetParam());

    if (arg.N == 0) 
        testing_getri_bad_arg<false,false,float>();

    arg.batch_count = 1;
    testing_getri<false,false,float>(arg);
}

TEST_P(GETRI, __double) {
    Arguments arg = setup_arguments(GetParam());

    if (arg.N == 0) 
        testing_getri_bad_arg<false,false,double>();

    arg.batch_count = 1;
    testing_getri<false,false,double>(arg);
}

TEST_P(GETRI, __float_complex) {
    Arguments arg = setup_arguments(GetParam());

    if (arg.N == 0)
        testing_getri_bad_arg<false,false,rocblas_float_complex>();

    arg.batch_count = 1;
    testing_getri<false,false,rocblas_float_complex>(arg);
}

TEST_P(GETRI, __double_complex) {
    Arguments arg = setup_arguments(GetParam());

    if (arg.N == 0)
        testing_getri_bad_arg<false,false,rocblas_double_complex>();

    arg.batch_count = 1;
    testing_getri<false,false,rocblas_double_complex>(arg);
}





INSTANTIATE_TEST_CASE_P(daily_lapack, GETRI,
                        ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_CASE_P(checkin_lapack, GETRI,
                        ValuesIn(matrix_size_range));
