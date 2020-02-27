/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgbr.hpp"
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


typedef std::tuple<vector<int>, vector<int>> orgbr_tuple;

// vector of vector, each vector is a {M or N, N or M, K};
const vector<vector<int>> size_range = {
    {-1,1,1}, {0,1,1}, {1,-1,1}, {1,0,1}, {1,1,-1}, {30,30,0}, {10,30,5}, {20,5,10}, {20,20,25}, {50,50,30}, {70,40,40}, {100,100,80}
};

// each is a {lda, st}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'
const vector<vector<int>> store = {
    {-1, 0}, {-1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

const vector<vector<int>> large_size_range = {
    {200,150,100}, {270,270,270}, {400,400,405}, {800,500,300}, {1500,1000,300}, {2024,2024,2030} 
};


Arguments setup_arguments_orgbr(orgbr_tuple tup) 
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.storev = store[1] == 1 ? 'R' : 'C';
    arg.K = size[2];
    if (store[1]) {
        arg.N = size[0];
        arg.lda = max(size[1],size[2]);
        arg.M = size[1];
    } else {
        arg.M = size[0];
        arg.lda = size[0];
        arg.N = size[1];
    }

    arg.lda += store[0]*10;

    arg.timing = 0;

    return arg;
}

class OrthoGen : public ::TestWithParam<orgbr_tuple> {
protected:
    OrthoGen() {}
    virtual ~OrthoGen() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(OrthoGen, orgbr_float) {
    Arguments arg = setup_arguments_orgbr(GetParam());

    rocblas_status status = testing_orgbr<float>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'C' && (arg.N > arg.M || arg.N < min(arg.M,arg.K))) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'R' && (arg.M > arg.N || arg.M < min(arg.N,arg.K))) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoGen, orgbr_double) {
    Arguments arg = setup_arguments_orgbr(GetParam());

    rocblas_status status = testing_orgbr<double>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'C' && (arg.N > arg.M || arg.N < min(arg.M,arg.K))) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'R' && (arg.M > arg.N || arg.M < min(arg.N,arg.K))) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, OrthoGen,
                        Combine(ValuesIn(large_size_range),
                                ValuesIn(store)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, OrthoGen,
                        Combine(ValuesIn(size_range),
                                ValuesIn(store)));
