/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_trtri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef std::tuple<vector<int>, vector<int>> trtri_tuple;

// each matrix_size_range vector is a {n, lda}

// each triangle_range is a {uplo, diag}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    {0, 1},             //quick return
    {-1, 1}, {20, 5},   //invalid
    {32, 32}, {50, 50}, {70, 100}, {100, 150}
};

const vector<vector<int>> triangle_range = {
    {0, 0}, {1, 0}, {0, 1}, {1, 1}
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {500, 600}, {640, 640}, {1000, 1024}, {1200, 1230} 
};


Arguments trtri_setup_arguments(trtri_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> tri = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];
    
    arg.uplo_option = tri[0] ? 'L' : 'U';
    arg.diag_option = tri[1] ? 'U' : 'N';

    arg.timing = 0;

    return arg;
}

class TRTRI : public ::TestWithParam<trtri_tuple> {
protected:
    TRTRI() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(TRTRI, __float) {
    Arguments arg = trtri_setup_arguments(GetParam());

    if (arg.N == 0 && arg.uplo_option == 'U') 
        testing_trtri_bad_arg<float>();

    testing_trtri<float>(arg);
}

TEST_P(TRTRI, __double) {
    Arguments arg = trtri_setup_arguments(GetParam());

    if (arg.N == 0 && arg.uplo_option == 'U') 
        testing_trtri_bad_arg<double>();

    testing_trtri<double>(arg);
}

TEST_P(TRTRI, __float_complex) {
    Arguments arg = trtri_setup_arguments(GetParam());

    if (arg.N == 0 && arg.uplo_option == 'U')
        testing_trtri_bad_arg<rocblas_float_complex>();

    testing_trtri<rocblas_float_complex>(arg);
}

TEST_P(TRTRI, __double_complex) {
    Arguments arg = trtri_setup_arguments(GetParam());

    if (arg.N == 0 && arg.uplo_option == 'U')
        testing_trtri_bad_arg<rocblas_double_complex>();

    testing_trtri<rocblas_double_complex>(arg);
}


INSTANTIATE_TEST_SUITE_P(daily_lapack, TRTRI,
                         Combine(ValuesIn(large_matrix_size_range),
                                 ValuesIn(triangle_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, TRTRI,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(triangle_range)));
