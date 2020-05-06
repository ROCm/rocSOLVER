/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef std::tuple<vector<int>, vector<int>> getrs_tuple;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, trans};
// if trans = 0 then no transpose
// if trans = 1 then transpose
// if trans = 2 then conjugate transpose

// case when N = nrhs = 0 will execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    {0, 1, 1},                              //quick return
    {-1, 1, 1}, {10, 2, 10}, {10, 10, 2},   //invalid 
    {20, 20, 20}, {30, 50, 30}, {30, 30, 50}, {50, 60, 60}
};
const vector<vector<int>> matrix_sizeB_range = {
    {0, 0},     //quick return
    {-1, 0},    //invalid 
    {10, 0}, {20, 1}, {30, 2},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_sizeA_range = {
    {70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}
};
const vector<vector<int>> large_matrix_sizeB_range = {
    {100, 0}, {150, 0}, {200, 1}, {524, 2}, {1000, 2},
};


Arguments setup_getrs_arguments(getrs_tuple tup) {
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_sizeA[0];
    arg.N = matrix_sizeB[0];
    arg.lda = matrix_sizeA[1];
    arg.ldb = matrix_sizeA[2];

    if (matrix_sizeB[1] == 0)
        arg.transA_option = 'N';
    else if(matrix_sizeB[1] == 1)
        arg.transA_option = 'T';
    else
        arg.transA_option = 'C';

    arg.timing = 0;

    return arg;
}

class GETRS : public ::TestWithParam<getrs_tuple> {
protected:
    GETRS() {}
    virtual ~GETRS() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};


// non-batch tests

TEST_P(GETRS, __float) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,false,float>();
//    testing_getrs<float,float>(arg);
}

TEST_P(GETRS, __double) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,false,double>();
//    testing_getrs<double,double>(arg);
}

TEST_P(GETRS, __float_complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,false,rocblas_float_complex>();
//    testing_getrs<rocblas_float_complex,float>(arg);
}

TEST_P(GETRS, __double_complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,false,rocblas_double_complex>();
//    testing_getrs<rocblas_double_complex,double>(arg);
}



// batched tests

TEST_P(GETRS, batched__float) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true,false,float>();
//    testing_getrs<float,float>(arg);
}

TEST_P(GETRS, batched__double) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true,false,double>();
//    testing_getrs<double,double>(arg);
}

TEST_P(GETRS, batched_float__complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true,false,rocblas_float_complex>();
//    testing_getrs<rocblas_float_complex,float>(arg);
}

TEST_P(GETRS, batched_double__complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<true,false,rocblas_double_complex>();
//    testing_getrs<rocblas_double_complex,double>(arg);
}



// strided_batched tests

TEST_P(GETRS, strided_batched__float) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,true,float>();
//    testing_getrs<float,float>(arg);
}

TEST_P(GETRS, strided_batched__double) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,true,double>();
//    testing_getrs<double,double>(arg);
}

TEST_P(GETRS, strided_batched__float_complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,true,rocblas_float_complex>();
//    testing_getrs<rocblas_float_complex,float>(arg);
}

TEST_P(GETRS, strided_batched__double_complex) {
    Arguments arg = setup_getrs_arguments(GetParam());
    if (arg.M == 0 && arg.N == 0)
        testing_getrs_bad_arg<false,true,rocblas_double_complex>();
//    testing_getrs<rocblas_double_complex,double>(arg);
}




// daily_lapack tests normal execution with medium to large sizes
INSTANTIATE_TEST_CASE_P(daily_lapack, GETRS,
                        Combine(ValuesIn(large_matrix_sizeA_range),
                                ValuesIn(large_matrix_sizeB_range)));

// checkin_lapack tests normal execution with small sizes, invalid sizes,
// quick returns, and corner cases
INSTANTIATE_TEST_CASE_P(checkin_lapack, GETRS,
                        Combine(ValuesIn(matrix_sizeA_range),
                                ValuesIn(matrix_sizeB_range)));
