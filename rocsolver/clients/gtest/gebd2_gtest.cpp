/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gebd2.hpp"
#include "utility.h"
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef std::tuple<vector<int>, int> gebd_tuple;

// vector of vector, each vector is a {M, lda};
const vector<vector<int>> matrix_size_range = {
    {0, 1}, {-1, 1}, {20, 5}, {50, 50}, {70, 100}, {130, 130}, {150, 200}
};

// each is a N
const vector<int> n_size_range = {
    -1, 0, 16, 20, 130, 150
};

const vector<vector<int>> large_matrix_size_range = {
    {152, 152}, {640, 640}, {1000, 1024}, 
};

const vector<int> large_n_size_range = {
    64, 98, 130, 220, 400
};


Arguments setup_arguments_bd(gebd_tuple tup) 
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = n_size;
    arg.lda = matrix_size[1];

    arg.timing = 0;

    return arg;
}

class Bidiag : public ::TestWithParam<gebd_tuple> {
protected:
    Bidiag() {}
    virtual ~Bidiag() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(Bidiag, gebd2_float) {
    Arguments arg = setup_arguments_bd(GetParam());

    rocblas_status status = testing_gebd2<float,float>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(Bidiag, gebd2_double) {
    Arguments arg = setup_arguments_bd(GetParam());

    rocblas_status status = testing_gebd2<double,double>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(Bidiag, gebd2_float_complex) {
    Arguments arg = setup_arguments_bd(GetParam());

    rocblas_status status = testing_gebd2<rocblas_float_complex,float>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(Bidiag, gebd2_double_complex) {
    Arguments arg = setup_arguments_bd(GetParam());

    rocblas_status status = testing_gebd2<rocblas_double_complex,double>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, Bidiag,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, Bidiag,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(n_size_range)));
