/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgl2_orglq.hpp"
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


typedef std::tuple<vector<int>, int> orglq_tuple;

// vector of vector, each vector is a {M, lda, K};
const vector<vector<int>> m_size_range = {
    {0, 1, 1}, {-1, 1, 1}, {20, 5, 1}, {10, 10, 20}, {10, 10, 10}, {20, 50, 20}, {130, 130, 50}
};

// each is a N
const vector<int> n_size_range = {
    -1, 0, 50, 70, 130
};

const vector<vector<int>> large_m_size_range = {
    {164, 164, 130}, {198, 640, 198}, {130, 130, 130}, {220, 220, 140}, {400, 400, 200} 
};

const vector<int> large_n_size_range = {
    130, 400, 640, 1000, 2000
};


Arguments setup_arguments_orglq(orglq_tuple tup) 
{
    vector<int> m_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = m_size[0];
    arg.N = n_size;
    arg.K = m_size[2];
    arg.lda = m_size[1];

    arg.timing = 0;

    return arg;
}

class OrthoRowGen : public ::TestWithParam<orglq_tuple> {
protected:
    OrthoRowGen() {}
    virtual ~OrthoRowGen() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(OrthoRowGen, orgl2_float) {
    Arguments arg = setup_arguments_orglq(GetParam());

    rocblas_status status = testing_orgl2_orglq<float,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N < arg.M || arg.K > arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoRowGen, orgl2_double) {
    Arguments arg = setup_arguments_orglq(GetParam());

    rocblas_status status = testing_orgl2_orglq<double,0>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N < arg.M || arg.K > arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoRowGen, orglq_float) {
    Arguments arg = setup_arguments_orglq(GetParam());

    rocblas_status status = testing_orgl2_orglq<float,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N < arg.M || arg.K > arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}

TEST_P(OrthoRowGen, orglq_double) {
    Arguments arg = setup_arguments_orglq(GetParam());

    rocblas_status status = testing_orgl2_orglq<double,1>(arg);

    // if not success, then the input argument is problematic, so detect the error
    // message
    if (status != rocblas_status_success) {
        if (arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.N < arg.M || arg.K > arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        } else if (arg.lda < arg.M) {
            EXPECT_EQ(rocblas_status_invalid_size, status);
        }
    }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, OrthoRowGen,
                        Combine(ValuesIn(large_m_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, OrthoRowGen,
                        Combine(ValuesIn(m_size_range),
                                ValuesIn(n_size_range)));
