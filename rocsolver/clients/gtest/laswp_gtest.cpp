/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_laswp.hpp"
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


typedef std::tuple<vector<int>, vector<int>> wpTuple;

//{n,lda}
const vector<vector<int>> range1 = {
    {-1,1}, {1,0},                  //error 
    {0,1},                          //quick return
    {10,100}, {20,100}, {30,100}    
};

//{k1,k2,inc}
const vector<vector<int>> range2 = {
    {0,1,1}, {1,0,1}, {1,2,0}, {2,1,1},     //error 
    {1,2,1}, {3,5,2}, {5,10,-1}, {3,12,-2}
};

const vector<vector<int>> large_range1 = {
    {192,100}, {250,100}, {500,100}, {1500,100}
};

const vector<vector<int>> large_range2 = {
    {1,50,1}, {5,60,2}, {3,70,-1}, {20,100,-2}
};


Arguments laswp_setup_arguments(wpTuple tup) {

  vector<int> matrix_size = std::get<0>(tup);
  vector<int> pivots = std::get<1>(tup);
      
  Arguments arg;

  arg.N = matrix_size[0];
  arg.lda = matrix_size[1];
  arg.k1 = pivots[0];
  arg.k2 = pivots[1];
  arg.incx = pivots[2];
  arg.timing = 0;

  return arg;
}

class permut : public ::TestWithParam<wpTuple> {
protected:
  permut() {}
  virtual ~permut() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(permut, laswp_float) {
  Arguments arg = laswp_setup_arguments(GetParam());

  rocblas_status status = testing_laswp<float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.lda < 1 || arg.k1 < 1 || arg.k2 < 1 || arg.k1 < arg.k2) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

TEST_P(permut, laswp_double) {
  Arguments arg = laswp_setup_arguments(GetParam());

  rocblas_status status = testing_laswp<double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.lda < 1 || arg.k1 < 1 || arg.k2 < 1 || arg.k1 < arg.k2) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

TEST_P(permut, laswp_float_complex) {
  Arguments arg = laswp_setup_arguments(GetParam());

  rocblas_status status = testing_laswp<rocblas_float_complex>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.lda < 1 || arg.k1 < 1 || arg.k2 < 1 || arg.k1 < arg.k2) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

TEST_P(permut, laswp_double_complex) {
  Arguments arg = laswp_setup_arguments(GetParam());

  rocblas_status status = testing_laswp<rocblas_double_complex>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0 || arg.lda < 1 || arg.k1 < 1 || arg.k2 < 1 || arg.k1 < arg.k2) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, permut,
                        Combine(ValuesIn(large_range1),
                                ValuesIn(large_range2)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, permut,
                        Combine(ValuesIn(range1),
                                ValuesIn(range2)));
