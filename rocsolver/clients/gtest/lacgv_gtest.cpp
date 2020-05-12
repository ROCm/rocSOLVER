/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_lacgv.hpp"
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


typedef vector<int> wpTuple;

//{n,inc}
const vector<vector<int>> range = {
    {-1,1}, {1,0},                  //error 
    {0,1},                          //quick return
    {10,1}, {10,-1}, {20,2}, {30,3}, {30,-3}
};

const vector<vector<int>> large_range = {
    {192,10}, {192,-10}, {250,20}, {500,30}, {1500,40}, {1500,-40}
};


Arguments lacgv_setup_arguments(wpTuple tup) {

  Arguments arg;

  arg.N = tup[0];
  arg.incx = tup[1];

  return arg;
}

class conjg : public ::TestWithParam<wpTuple> {
protected:
  conjg() {}
  virtual ~conjg() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(conjg, lacgv_float_complex) {
  Arguments arg = lacgv_setup_arguments(GetParam());

  rocblas_status status = testing_lacgv<rocblas_float_complex>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || !arg.incx) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

TEST_P(conjg, lacgv_double_complex) {
  Arguments arg = lacgv_setup_arguments(GetParam());

  rocblas_status status = testing_lacgv<rocblas_double_complex>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || !arg.incx) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, conjg,
                        ValuesIn(large_range));

INSTANTIATE_TEST_CASE_P(checkin_lapack, conjg,
                        ValuesIn(range));
