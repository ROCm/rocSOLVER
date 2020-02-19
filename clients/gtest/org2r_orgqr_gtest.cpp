/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_org2r_orgqr.hpp"
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


typedef std::tuple<vector<int>, vector<int>> orgqr_tuple;

// vector of vector, each vector is a {M, lda};
const vector<vector<int>> m_size_range = {
    {0, 1}, {-1, 1}, {20, 5}, {50, 50}, {70, 100}, {130, 130}
};

// each is a {N, K}
const vector<vector<int>> n_size_range = {
    {-1, 1}, {0, 1}, {1, -1}, {1, 0}, {10, 20}, {20, 20}, {130, 130}
};

const vector<vector<int>> large_m_size_range = {
    {152, 152}, {640, 640}, {1000, 1024}, {2000, 2000} 
};

const vector<vector<int>> large_n_size_range = {
    {164, 162}, {198, 140}, {130, 130}, {220, 220}, {400, 200}
};


Arguments setup_arguments_org(orgqr_tuple tup) 
{
    vector<int> m_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = m_size[0];
    arg.N = n_size[0];
    arg.K = n_size[1];
    arg.lda = m_size[1];

    arg.timing = 0;

    return arg;
}

class OrthoMaxGen : public ::TestWithParam<orgqr_tuple> {
protected:
    OrthoMaxGen() {}
    virtual ~OrthoMaxGen() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(OrthoMaxGen, org2r_float) {
    Arguments arg = setup_arguments_org(GetParam());

    rocblas_status status = testing_org2r_orgqr<float,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N > arg.M || arg.K > arg.N) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoMaxGen, org2r_double) {
    Arguments arg = setup_arguments_org(GetParam());

    rocblas_status status = testing_org2r_orgqr<double,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N > arg.M || arg.K > arg.N) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoMaxGen, orgqr_float) {
    Arguments arg = setup_arguments_org(GetParam());

    rocblas_status status = testing_org2r_orgqr<float,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N > arg.M || arg.K > arg.N) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoMaxGen, orgqr_double) {
    Arguments arg = setup_arguments_org(GetParam());

    rocblas_status status = testing_org2r_orgqr<double,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N > arg.M || arg.K > arg.N) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, OrthoMaxGen,
                        Combine(ValuesIn(large_m_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, OrthoMaxGen,
                        Combine(ValuesIn(m_size_range),
                                ValuesIn(n_size_range)));
