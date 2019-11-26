/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_geqr2_geqrf_batched.hpp"
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


typedef std::tuple<vector<int>, vector<int>> qr_tuple;

// **** ONLY TESTING NORMNAL USE CASES
//      I.E. WHEN STRIDEA >= LDA*N AND STRIDEP >= MIN(M,N) ****

// vector of vector, each vector is a {M, lda};
const vector<vector<int>> matrix_size_range = {
    {0, 1}, {-1, 1}, {20, 5}, {50, 50}, {70, 100}, {130, 130}, {150, 200}
};

// each is a {N, stP}
// if stP == 0: stridep is min(M,N)
// if stP == 1: stridep > min(M,N)
const vector<vector<int>> n_size_range = {
    {-1, 0}, {0, 0}, {16, 0}, {20, 1}, {130, 0}, {150, 1}    
};

const vector<vector<int>> large_matrix_size_range = {
    {152, 152}, {640, 640}, {1000, 1024}
};

const vector<vector<int>> large_n_size_range = {
    {64, 0}, {98, 0}, {130, 0}, {220, 1}, {400, 0}, 
};


Arguments setup_arguments_qrb(qr_tuple tup) 
{
  vector<int> matrix_size = std::get<0>(tup);
  vector<int> n_size = std::get<1>(tup);
  Arguments arg;

  arg.M = matrix_size[0];
  arg.N = n_size[0];
  arg.lda = matrix_size[1];

  arg.bsp = min(arg.M, arg.N) + n_size[1]; 

  arg.timing = 0;
  arg.batch_count = 3;
  return arg;
}

class QRfact_b : public ::TestWithParam<qr_tuple> {
protected:
  QRfact_b() {}
  virtual ~QRfact_b() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(QRfact_b, geqr2_batched_float) {
  Arguments arg = setup_arguments_qrb(GetParam());

  rocblas_status status = testing_geqr2_geqrf_batched<float,0>(arg);

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

TEST_P(QRfact_b, geqr2_batched_double) {
  Arguments arg = setup_arguments_qrb(GetParam());

  rocblas_status status = testing_geqr2_geqrf_batched<double,0>(arg);

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

TEST_P(QRfact_b, geqrf_batched_float) {
  Arguments arg = setup_arguments_qrb(GetParam());

  rocblas_status status = testing_geqr2_geqrf_batched<float,1>(arg);

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

TEST_P(QRfact_b, geqrf_batched_double) {
  Arguments arg = setup_arguments_qrb(GetParam());

  rocblas_status status = testing_geqr2_geqrf_batched<double,1>(arg);

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


INSTANTIATE_TEST_CASE_P(daily_lapack, QRfact_b,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, QRfact_b,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(n_size_range)));


