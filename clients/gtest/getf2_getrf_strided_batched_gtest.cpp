/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2_getrf_strided_batched.hpp"
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


typedef std::tuple<vector<int>, vector<int>> getf2_getrf_tuple;

// **** ONLY TESTING NORMNAL USE CASES
//      I.E. WHEN STRIDEA >= LDA*N AND STRIDEP >= MIN(M,N) ****


// vector of vector, each vector is a {M, lda, stA};
// if stA == 0: strideA is lda*N
// if stA == 1: strideA > lda*N 
const vector<vector<int>> matrix_size_range = {
    {0, 1, 0}, {-1, 1, 0}, {20, 5, 0}, {50, 50, 1}, {70, 100, 0}
};

// each is a {N, stP}
// if stP == 0: stridep is min(M,N)
// if stP == 1: stridep > min(M,N)
const vector<vector<int>> n_size_range = {
    {-1, 0}, {0, 0}, {20, 0}, {40, 1}, {100, 0}
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 1}, {640, 640, 0}, {1000, 1024, 0}, 
};

const vector<vector<int>> large_n_size_range = {
    {45, 1}, {64, 0}, {520, 0}, {1000, 0}, {1024, 0}, 
};


Arguments setup_arguments(getf2_getrf_tuple tup) 
{
  vector<int> matrix_size = std::get<0>(tup);
  vector<int> n_size = std::get<1>(tup);
  Arguments arg;

  arg.M = matrix_size[0];
  arg.N = n_size[0];
  arg.lda = matrix_size[1];

  arg.bsp = min(arg.M, arg.N) + n_size[1]; 
  arg.bsa = arg.lda * arg.N + matrix_size[2];

  arg.timing = 0;
  arg.batch_count = 3;
  return arg;
}

class LUfact_sb : public ::TestWithParam<getf2_getrf_tuple> {
protected:
  LUfact_sb() {}
  virtual ~LUfact_sb() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(LUfact_sb, getf2_strided_batched_float) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf_strided_batched<float,0>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else {
      cerr << "unknown error...";
      EXPECT_EQ(1000, status);
    }
  }
}

TEST_P(LUfact_sb, getf2_strided_batched_double) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf_strided_batched<double,0>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else {
      cerr << "unknown error...";
      EXPECT_EQ(1000, status);
    }
  }
}

TEST_P(LUfact_sb, getrf_strided_batched_float) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf_strided_batched<float,1>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else {
      cerr << "unknown error...";
      EXPECT_EQ(1000, status);
    }
  }
}

TEST_P(LUfact_sb, getrf_strided_batched_double) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf_strided_batched<double,1>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else {
      cerr << "unknown error...";
      EXPECT_EQ(1000, status);
    }
  }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, LUfact_sb,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, LUfact_sb,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(n_size_range)));
