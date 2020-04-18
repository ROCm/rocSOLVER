/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_ormbr.hpp"
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


typedef std::tuple<vector<int>, vector<int>> ormbr_tuple;

// vector of vector, each vector is a {M, N, K};
const vector<vector<int>> size_range = {
    {-1,1,1}, {0,1,1}, {1,-1,1}, {1,0,1}, {1,1,-1}, {1,1,0}, 
    {10,30,5}, {20,5,10}, {20,20,25}, {50,50,30}, {70,40,40}, 
};

// each is a {lda, ldc, s, t, st}
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
// if st = 0, then storev = 'C'
// if st = 1, then storev = 'R'
const vector<vector<int>> store = {
    {-1, 0, 0, 0, 0}, {0, -1, 0, 0, 0}, 
    {1, 1, 0, 0, 0}, {1, 1, 0, 0, 1}, 
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 1, 1},
    {0, 0, 1, 0, 0},
    {0, 0, 1, 0, 1},
    {0, 0, 1, 1, 0},
    {0, 0, 1, 1, 1},
};

const vector<vector<int>> large_size_range = {
    {200,150,100}, {270,270,270}, {400,400,405}, {800,500,300}, {1500,1000,300},  
};


Arguments setup_arguments_ormbr(ormbr_tuple tup) 
{
    vector<int> size = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.storev = store[4] == 0 ? 'C' : 'R';
    arg.transA_option = store[3] == 0 ? 'N' : 'T';
    arg.side_option = store[2] == 0 ? 'L' : 'R';

    arg.K = size[2];
    arg.N = size[1];
    arg.M = size[0];

    arg.ldc = arg.M + store[1]*10;

    int nq = arg.side_option == 'L' ? arg.M : arg.N;
    if (arg.storev == 'C') {
        arg.lda = nq;
    } else {
        arg.lda = min(nq,arg.K);
    }

    arg.lda += store[0]*10;

    arg.timing = 0;

    return arg;
}

class OrthoApp : public ::TestWithParam<ormbr_tuple> {
protected:
    OrthoApp() {}
    virtual ~OrthoApp() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(OrthoApp, ormbr_float) {
    Arguments arg = setup_arguments_ormbr(GetParam());

    rocblas_status status = testing_ormbr<float>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        int nq = arg.side_option == 'L' ? arg.M : arg.N;
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'C' && arg.lda < nq) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'R' && arg.lda < min(nq,arg.K)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoApp, ormbr_double) {
    Arguments arg = setup_arguments_ormbr(GetParam());

    rocblas_status status = testing_ormbr<double>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        int nq = arg.side_option == 'L' ? arg.M : arg.N;
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.ldc < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'C' && arg.lda < nq) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.storev == 'R' && arg.lda < min(nq,arg.K)) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, OrthoApp,
                        Combine(ValuesIn(large_size_range),
                                ValuesIn(store)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, OrthoApp,
                        Combine(ValuesIn(size_range),
                                ValuesIn(store)));
