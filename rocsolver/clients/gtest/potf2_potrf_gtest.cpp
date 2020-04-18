/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potf2_potrf.hpp"
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


typedef std::tuple<vector<int>, char> chol_tuple;

// vector of vector, each vector is a {N, lda};
const vector<vector<int>> matrix_size_range = {
    {-1, 1}, {0, 1}, {10, 2}, {10, 10}, {20, 30}, {50, 50}, {70, 80}
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {640, 960}, {1000, 1000}, {1024, 1024}, {2000, 2000},
};

// vector of char, each is an uplo, which can be "Lower (L) or Upper (U)"
// Each letter is capitalizied, e.g. do not use 'l', but use 'L' instead.

const vector<char> uplo_range = {'L', 'U'};

Arguments setup_chol_arguments(chol_tuple tup) 
{
  vector<int> matrix_size = std::get<0>(tup);
  char uplo = std::get<1>(tup);

  Arguments arg;

  // see the comments about matrix_size_range above
  arg.N = matrix_size[0];
  arg.lda = matrix_size[1];

  arg.uplo_option = uplo;

  arg.timing = 0;

  return arg;
}

class CholeskyFact : public ::TestWithParam<chol_tuple> {
protected:
  CholeskyFact() {}
  virtual ~CholeskyFact() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(CholeskyFact, potf2_float) {
  Arguments arg = setup_chol_arguments(GetParam());

  rocblas_status status = testing_potf2_potrf<float,0>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.N) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(CholeskyFact, potf2_double) {
  Arguments arg = setup_chol_arguments(GetParam());

  rocblas_status status = testing_potf2_potrf<double,0>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.N) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(CholeskyFact, potrf_float) {
  Arguments arg = setup_chol_arguments(GetParam());

  rocblas_status status = testing_potf2_potrf<float,1>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.N) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(CholeskyFact, potrf_double) {
  Arguments arg = setup_chol_arguments(GetParam());

  rocblas_status status = testing_potf2_potrf<double,1>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.N) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}


INSTANTIATE_TEST_CASE_P(daily_lapack, CholeskyFact,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(uplo_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, CholeskyFact,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(uplo_range)));
