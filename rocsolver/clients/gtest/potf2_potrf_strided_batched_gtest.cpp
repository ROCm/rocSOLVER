/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potf2_potrf_strided_batched.hpp"
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


typedef std::tuple<vector<int>, char> cholSB_tuple;

// **** ONLY TESTING NORMNAL USE CASES
//      I.E. WHEN STRIDEA >= LDA*N ****

// vector of vector, each vector is a {N, lda, stA};
// if stA == 0: strideA is lda*N
// if stA == 1; strideA > lda*N
const vector<vector<int>> matrix_size_range = {
    {-1, 1, 0}, {0, 1, 0}, {10, 2, 0}, {10, 10, 0}, {20, 30, 1}, {50, 50, 0}, {70, 80, 0}
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0}, {640, 960, 1}, {1000, 1000, 0}, {1024, 1024, 1}, {2000, 2000, 0},
};

// vector of char, each is an uplo, which can be "Lower (L) or Upper (U)"
// Each letter is capitalizied, e.g. do not use 'l', but use 'L' instead.

const vector<char> uplo_range = {'L', 'U'};

Arguments setup_chol_arguments_sb(cholSB_tuple tup) 
{
  vector<int> matrix_size = std::get<0>(tup);
  char uplo = std::get<1>(tup);

  Arguments arg;

  // see the comments about matrix_size_range above
  arg.N = matrix_size[0];
  arg.lda = matrix_size[1];
  arg.bsa = arg.N * arg.lda + matrix_size[2];

  arg.uplo_option = uplo;

  arg.timing = 0;
  arg.batch_count = 3;

  return arg;
}

class CholeskyFact_sb : public ::TestWithParam<cholSB_tuple> {
protected:
  CholeskyFact_sb() {}
  virtual ~CholeskyFact_sb() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(CholeskyFact_sb, potf2_strided_batched_float) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<float,float,0>(arg);

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

TEST_P(CholeskyFact_sb, potf2_strided_batched_double) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<double,double,0>(arg);

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

TEST_P(CholeskyFact_sb, potf2_strided_batched_float_complex) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<rocblas_float_complex,float,0>(arg);

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

TEST_P(CholeskyFact_sb, potf2_strided_batched_double_complex) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<rocblas_double_complex,double,0>(arg);

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

TEST_P(CholeskyFact_sb, potrf_strided_batched_float) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<float,float,1>(arg);

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

TEST_P(CholeskyFact_sb, potrf_strided_batched_double) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<double,double,1>(arg);

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

TEST_P(CholeskyFact_sb, potrf_strided_batched_float_complex) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<rocblas_float_complex,float,1>(arg);

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

TEST_P(CholeskyFact_sb, potrf_strided_batched_double_complex) {
  Arguments arg = setup_chol_arguments_sb(GetParam());

  rocblas_status status = testing_potf2_potrf_strided_batched<rocblas_double_complex,double,1>(arg);

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


INSTANTIATE_TEST_CASE_P(daily_lapack, CholeskyFact_sb,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(uplo_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, CholeskyFact_sb,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(uplo_range)));
