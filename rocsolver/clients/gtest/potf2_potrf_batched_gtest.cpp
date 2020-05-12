/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potf2_potrf_batched.hpp"
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


typedef std::tuple<vector<int>, char> cholB_tuple;

// **** ONLY TESTING NORMNAL USE CASES
//      I.E. WHEN STRIDEA >= LDA*N ****

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

Arguments setup_chol_arguments_b(cholB_tuple tup) 
{
  vector<int> matrix_size = std::get<0>(tup);
  char uplo = std::get<1>(tup);

  Arguments arg;

  // see the comments about matrix_size_range above
  arg.N = matrix_size[0];
  arg.lda = matrix_size[1];

  arg.uplo_option = uplo;

  arg.timing = 0;
  arg.batch_count = 3;

  return arg;
}

class CholeskyFact_b : public ::TestWithParam<cholB_tuple> {
protected:
  CholeskyFact_b() {}
  virtual ~CholeskyFact_b() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(CholeskyFact_b, potf2_batched_float) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<float,float,0>(arg);

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

TEST_P(CholeskyFact_b, potf2_batched_double) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<double,double,0>(arg);

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

TEST_P(CholeskyFact_b, potf2_batched_float_complex) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<rocblas_float_complex,float,0>(arg);

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

TEST_P(CholeskyFact_b, potf2_batched_double_complex) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<rocblas_double_complex,double,0>(arg);

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

TEST_P(CholeskyFact_b, potrf_batched_float) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<float,float,1>(arg);

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

TEST_P(CholeskyFact_b, potrf_batched_double) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<double,double,1>(arg);

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

TEST_P(CholeskyFact_b, potrf_batched_float_complex) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<rocblas_float_complex,float,1>(arg);

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

TEST_P(CholeskyFact_b, potrf_batched_double_complex) {
  Arguments arg = setup_chol_arguments_b(GetParam());

  rocblas_status status = testing_potf2_potrf_batched<rocblas_double_complex,double,1>(arg);

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


INSTANTIATE_TEST_CASE_P(daily_lapack, CholeskyFact_b,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(uplo_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, CholeskyFact_b,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(uplo_range)));
