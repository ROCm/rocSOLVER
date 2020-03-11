/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orm2r_ormqr.hpp"
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


typedef std::tuple<vector<int>, vector<int>> ormqr_tuple;

// vector of vector, each vector is a {M, N, K};
const vector<vector<int>> size_range = {
    {-1,1,1}, {0,1,1}, {1,-1,1}, {1,0,1}, {1,1,-1}, {30,30,0}, {20,10,20}, {15,25,25}, {40,40,50}, {45,40,40}   
};

// each is a {lda, ldc, s, t}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if ldc = -1, then ldc < limit (invalid size)
// if ldc = 0, then ldc = limit
// if ldc = 1, then ldc > limit
// if s = 0, then side = 'L'
// if s = 1, then side = 'R'
// if t = 0, then trans = 'N'
// if t = 1, then trans = 'T'
const vector<vector<int>> op_range = {
    {-1,0,0,0}, {0,-1,0,0}, {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1}, {1,1,0,0}   
};

const vector<vector<int>> large_size_range = {
    {100,100,100}, {150,100,80}, {300,400,300}, {1024,1000,950}, {1500,1500,1000}
};


Arguments setup_arguments_orm(ormqr_tuple tup) 
{
    vector<int> size = std::get<0>(tup);
    vector<int> op = std::get<1>(tup);

    Arguments arg;

    arg.M = size[0];
    arg.N = size[1];
    arg.K = size[2];
    arg.ldc = arg.M + op[1]*10;

    arg.transA_option = op[3] == 0 ? 'N' : 'T';
    arg.side_option = op[2] == 0 ? 'L' : 'R';

    if (op[2]) {
        arg.lda = arg.N + op[0]*10;
    } else {
        arg.lda = arg.M + op[0]*10;
    }

    arg.timing = 0;

    return arg;
}

class OrthoColApp : public ::TestWithParam<ormqr_tuple> {
protected:
    OrthoColApp() {}
    virtual ~OrthoColApp() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(OrthoColApp, orm2r_float) {
    Arguments arg = setup_arguments_orm(GetParam());

    rocblas_status status = testing_orm2r_ormqr<float,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'L' && (arg.K > arg.M || arg.lda < arg.M)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'R' && (arg.K > arg.N || arg.lda < arg.N)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoColApp, orm2r_double) {
    Arguments arg = setup_arguments_orm(GetParam());

    rocblas_status status = testing_orm2r_ormqr<double,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'L' && (arg.K > arg.M || arg.lda < arg.M)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'R' && (arg.K > arg.N || arg.lda < arg.N)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoColApp, ormqr_float) {
    Arguments arg = setup_arguments_orm(GetParam());

    rocblas_status status = testing_orm2r_ormqr<float,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'L' && (arg.K > arg.M || arg.lda < arg.M)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'R' && (arg.K > arg.N || arg.lda < arg.N)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoColApp, ormqr_double) {
    Arguments arg = setup_arguments_orm(GetParam());

    rocblas_status status = testing_orm2r_ormqr<double,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'L' && (arg.K > arg.M || arg.lda < arg.M)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.side_option == 'R' && (arg.K > arg.N || arg.lda < arg.N)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, OrthoColApp,
                        Combine(ValuesIn(large_size_range),
                                ValuesIn(op_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, OrthoColApp,
                        Combine(ValuesIn(size_range),
                                ValuesIn(op_range)));
