/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrs_batched.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;


typedef std::tuple<vector<int>, vector<int>> getrsB_tuple;

// **** THIS FUNCTION ONLY TESTS NORMNAL USE CASE
//      I.E. WHEN STRIDEP >= M ****

// vector of vector, each vector is a {N, lda, ldb};
// add/delete as a group
const vector<vector<int>> matrix_sizeA_range = {
    {-1, 1, 1}, {0, 1, 1}, {10, 2, 10}, {10, 10, 2}, {20, 20, 20}, {30, 50, 30}, {30, 30, 50}, {50, 60, 60}
};

// vector of vector, each vector is a {nrhs, trans, std};
// if trans = 0 then no transpose
// if trans = 1 then transpose
// if std = 0 strides are the minimum
// if std = 1 strides are larger
const vector<vector<int>> matrix_sizeB_range = {
    {-1, 0, 0}, {0, 0, 0}, {10, 0, 0}, {20, 1, 1}, {30, 0, 0},
};

const vector<vector<int>> large_matrix_sizeA_range = {
    {70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}
};

const vector<vector<int>> large_matrix_sizeB_range = {
    {100, 0, 0}, {150, 0, 1}, {200, 1, 0}, {524, 1, 1}, {1000, 0, 0},
};


Arguments setup_getrsB_arguments(getrsB_tuple tup) {

  vector<int> matrix_sizeA = std::get<0>(tup);
  vector<int> matrix_sizeB = std::get<1>(tup);

  Arguments arg;

  // see the comments about matrix_size_range above
  arg.M = matrix_sizeA[0];
  arg.N = matrix_sizeB[0];
  arg.lda = matrix_sizeA[1];
  arg.ldb = matrix_sizeA[2];

  if (matrix_sizeB[1] == 0)
    arg.transA_option = 'N';
  else
    arg.transA_option = 'T';

  arg.bsp = arg.M + matrix_sizeB[2]*10;

  arg.batch_count = 3;
  arg.timing = 0;

  return arg;
}

class LUsolver_b : public ::TestWithParam<getrsB_tuple> {
protected:
  LUsolver_b() {}
  virtual ~LUsolver_b() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(LUsolver_b, getrs_batched_float) {
  Arguments arg = setup_getrsB_arguments(GetParam());

  testing_getrs_batched<float,float>(arg);

}

TEST_P(LUsolver_b, getrs_batched_double) {
  Arguments arg = setup_getrsB_arguments(GetParam());

  testing_getrs_batched<double,double>(arg);

}

TEST_P(LUsolver_b, getrs_batched_float_complex) {
  Arguments arg = setup_getrsB_arguments(GetParam());

  testing_getrs_batched<rocblas_float_complex,float>(arg);

}

TEST_P(LUsolver_b, getrs_batched_double_complex) {
  Arguments arg = setup_getrsB_arguments(GetParam());

  testing_getrs_batched<rocblas_double_complex,double>(arg);

}

// This function mainly test the scope of matrix_size.
INSTANTIATE_TEST_CASE_P(daily_lapack, LUsolver_b,
                        Combine(ValuesIn(large_matrix_sizeA_range),
                                ValuesIn(large_matrix_sizeB_range)));

// THis function mainly test the scope of uplo_range, the scope of
// matrix_size_range is small
INSTANTIATE_TEST_CASE_P(checkin_lapack, LUsolver_b,
                        Combine(ValuesIn(matrix_sizeA_range),
                                ValuesIn(matrix_sizeB_range)));
