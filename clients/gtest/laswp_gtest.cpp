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

//TO BE IMPLEMENTED....

typedef std::tuple<vector<int>, vector<int>> wpTuple;

const vector<vector<int>> range1 = {
    {1,1}
};

const vector<vector<int>> range2 = {
    {-10,0}
};

const vector<vector<int>> large_range1 = {
    {192,192}
};

const vector<vector<int>> large_range2 = {
    {-10,0}
};


Arguments laswp_setup_arguments(wpTuple tup) {

  Arguments arg;

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

  //  if (arg.N < 0 || arg.K < 1 || arg.ldv < arg.N || arg.ldt < arg.K) {
  //    EXPECT_EQ(rocblas_status_invalid_size, status);
  //  } 
  }
}

TEST_P(permut, laswp_double) {
  Arguments arg = laswp_setup_arguments(GetParam());

  rocblas_status status = testing_laswp<double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

  //  if (arg.N < 0 || arg.K < 1 || arg.ldv < arg.N || arg.ldt < arg.K) {
  //    EXPECT_EQ(rocblas_status_invalid_size, status);
  //  } 
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, permut,
                        Combine(ValuesIn(large_range1),
                                ValuesIn(large_range2)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, permut,
                        Combine(ValuesIn(range1),
                                ValuesIn(range2)));
