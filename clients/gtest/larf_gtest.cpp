/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larf.hpp"
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


typedef std::tuple<vector<int>, vector<int>> mtuple;

//{M,N,lda}
const vector<vector<int>> matrix_size_range = {
    {-1,10,1}, {10,0,10}, {10,10,5}, {12,20,12}, {20,15,20}, {35,35,50}  
};

//{incx,s}
//if s = 0, then side = 'L'
//if s = 1, then side = 'R'
const vector<vector<int>> incx_range = {
    {-10,0}, {-5,1}, {-1,0}, {0,0}, {1,1}, {5,0}, {10,1}
};

//{M,N,lda}
const vector<vector<int>> large_matrix_size_range = {
    {192,192,192}, {640,300,700}, {1024,2000,1024}, {2547,2547,2550}
};



Arguments larf_setup_arguments(mtuple tup) {

  vector<int> matrix_size = std::get<0>(tup);
  vector<int> inc = std::get<1>(tup);

  Arguments arg;

  arg.M = matrix_size[0];
  arg.N = matrix_size[1];
  arg.lda = matrix_size[2];
  arg.incx = inc[0];

  arg.side_option = inc[1] == 1 ? 'R' : 'L';

  arg.timing = 0;

  return arg;
}

class app_HHreflec : public ::TestWithParam<mtuple> {
protected:
  app_HHreflec() {}
  virtual ~app_HHreflec() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(app_HHreflec, larf_float) {
  Arguments arg = larf_setup_arguments(GetParam());

  rocblas_status status = testing_larf<float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.M < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx == 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(app_HHreflec, larf_double) {
  Arguments arg = larf_setup_arguments(GetParam());

  rocblas_status status = testing_larf<double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.M < 0 || arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx == 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, app_HHreflec,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(incx_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, app_HHreflec,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(incx_range)));
