/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_larfb.hpp"
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


typedef std::tuple<vector<int>, vector<int>> bTuple;

//{M,N,lda,s,ldv,st}
//if s = 0, then side = 'L'
//if s = 1, then side = 'R'
//if st = 0, then storev = 'C'
//if st = 1, then storev = 'R'
const vector<vector<int>> matrix_size_range = {
    {-1,1,1,0,1,0}, {0,1,1,0,1,0}, {1,-1,1,0,1,0}, {1,0,1,0,1,0}, {15,15,5,0,15,0}, {12,5,12,0,5,0}, {5,12,15,1,5,0},
    {15,10,15,0,5,1},
    {15,15,15,0,15,0}, {18,20,20,1,20,0}, {20,18,20,0,20,0}, {20,30,20,1,30,0}, {50,35,50,0,50,0},
    {40,40,40,0,15,1}, {40,40,40,1,25,1}  
};

//{K,ldt,d,t}
//if d = 0, then direct = 'F'
//if d = 1, then direct = 'B'
//if t = 0, then trans = 'N'
//if t = 1, then trans = 'T'
const vector<vector<int>> reflector_size_range = {
    {0,1,0,0}, {5,1,0,0}, 
    {5,5,0,1}, {10,10,0,0}, {12,70,0,1}, {15,15,0,0}
};

const vector<vector<int>> large_matrix_size_range = {
    {192,192,192,0,192,0}, {640,640,640,1,700,0}, {640,640,700,0,640,0}, {840,1024,840,1,1024,0}, {2547,1980,2547,0,2547,0},
    {200,200,220,0,100,1}, {240,300,240,1,100,1}, {600,200,600,1,100,1}
};

const vector<vector<int>> large_reflector_size_range = {
    {35,35,0,1}, {50,70,0,0}, {85,85,0,1}, {100,150,0,0}
};


Arguments larfb_setup_arguments(bTuple tup) {

  vector<int> order_size = std::get<0>(tup);
  vector<int> reflector_size = std::get<1>(tup);

  Arguments arg;

  arg.M = order_size[0];
  arg.N = order_size[1];
  arg.lda = order_size[2];
  arg.ldv = order_size[4];
  
  arg.side_option = order_size[3] == 0 ? 'L' : 'R';  

  arg.K = reflector_size[0];
  arg.ldt = reflector_size[1];

  arg.direct_option = reflector_size[2] == 1 ? 'B' : 'F';
  arg.transH_option = reflector_size[3] == 1 ? 'T' : 'N';
  arg.storev = order_size[5] == 1 ? 'R' : 'C';

  arg.timing = 0;

  return arg;
}

class app_HHreflec_blk : public ::TestWithParam<bTuple> {
protected:
  app_HHreflec_blk() {}
  virtual ~app_HHreflec_blk() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(app_HHreflec_blk, larfb_float) {
  Arguments arg = larfb_setup_arguments(GetParam());

  rocblas_status status = testing_larfb<float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.K < 1 || arg.lda < arg.M || arg.ldt < arg.K) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
    else if (arg.storev == 'C' && ((arg.side_option == 'L' && arg.ldv < arg.M) || (arg.side_option == 'R' && arg.ldv < arg.N))) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
    else if (arg.storev == 'R' && arg.ldv < arg.K) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(app_HHreflec_blk, larfb_double) {
  Arguments arg = larfb_setup_arguments(GetParam());

  rocblas_status status = testing_larfb<double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {
    if (arg.M < 0 || arg.N < 0 || arg.K < 1 || arg.lda < arg.M || arg.ldt < arg.K) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } 
    else if (arg.storev == 'C' && ((arg.side_option == 'L' && arg.ldv < arg.M) || (arg.side_option == 'R' && arg.ldv < arg.N))) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
    else if (arg.storev == 'R' && arg.ldv < arg.K) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

INSTANTIATE_TEST_CASE_P(daily_lapack, app_HHreflec_blk,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_reflector_size_range)));

INSTANTIATE_TEST_CASE_P(checkin_lapack, app_HHreflec_blk,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(reflector_size_range)));
