/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larfg.hpp"
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


typedef std::tuple<int, int> mtuple;

const vector<int> n_size_range = {
    -1, 0, 1, 12, 20, 35,  
};

const vector<int> incx_range = {
    -1, 0, 1, 5, 8, 10,
};

const vector<int> large_n_size_range = {
    192, 640, 1024, 2547,
};



Arguments setup_arguments(mtuple tup) {

  int n_size = std::get<0>(tup);
  int inc = std::get<1>(tup);

  Arguments arg;

  arg.N = n_size;
  arg.incx = inc;

  arg.timing = 0;

  return arg;
}

class HHreflec : public ::TestWithParam<mtuple> {
protected:
  HHreflec() {}
  virtual ~HHreflec() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(HHreflec, larfg_float) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_larfg<float,float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx < 1) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(HHreflec, larfg_double) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_larfg<double,double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx < 1) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(HHreflec, larfg_float_complex) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_larfg<rocblas_float_complex,float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx < 1) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(HHreflec, larfg_double_complex) {
  Arguments arg = setup_arguments(GetParam());

  rocblas_status status = testing_larfg<rocblas_double_complex,double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.incx < 1) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, HHreflec,
                        Combine(ValuesIn(large_n_size_range),
                                ValuesIn(incx_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, HHreflec,
                        Combine(ValuesIn(n_size_range),
                                ValuesIn(incx_range)));
