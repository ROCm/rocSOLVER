/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2_getrf.hpp"
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


typedef std::tuple<vector<int>, int> getf2_getrf_tuple;

// vector of vector, each vector is a {M, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {0, 1}, {-1, 1}, {20, 5}, {32, 32}, {50, 50}, {70, 100}
};

// each is a N
const vector<int> n_size_range = {
    -1, 0, 16, 20, 40, 100,
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {640, 640}, {1000, 1024}, {2547, 2547},
};

const vector<int> large_n_size_range = {
    45, 64, 520, 1024, 2000, 
};


Arguments setup_arguments(getf2_getrf_tuple tup) {

  vector<int> matrix_size = std::get<0>(tup);
  int n_size = std::get<1>(tup);

  Arguments arg;

  arg.M = matrix_size[0];
  arg.N = n_size;
  arg.lda = matrix_size[1];

  arg.timing = 0;

  return arg;
}

class LUfact : public ::TestWithParam<getf2_getrf_tuple> {
protected:
  LUfact() {}
  virtual ~LUfact() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(LUfact, getf2_float) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf<float,0>(arg);

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

TEST_P(LUfact, getf2_double) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf<double,0>(arg);

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

TEST_P(LUfact, getrf_float) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf<float,1>(arg);

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

TEST_P(LUfact, getrf_double) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_getf2_getrf<double,1>(arg);

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


INSTANTIATE_TEST_CASE_P(daily_lapack, LUfact,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, LUfact,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(n_size_range)));
