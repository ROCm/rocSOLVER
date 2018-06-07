/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2.hpp"
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

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,
// std::tuple is good enough;

typedef std::tuple<vector<int>, int> getf2_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one
which invalidates the matrix. like lda pairs with M, and "lda must >= M". case
"lda < M" will be guarded by argument-checkers inside API of course. Yet, the
goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not
necessary
=================================================================== */

// vector of vector, each vector is a {M, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {-1, 1}, {10, 10}, {10, 20}, {500, 500}, {500, 750},
};

// each is a N
const vector<int> n_size_range = {
    1, 20, 40, 600, 600,
};

const vector<vector<int>> large_matrix_size_range = {
    {192, 192}, {640, 640}, {1000, 1000}, {1024, 1024}, {2000, 2000},
};

const vector<int> large_n_size_range = {
    192, 640, 1000, 1024, 2000,
};

/* ===============Google Unit
 * Test==================================================== */

/* =====================================================================
     LAPACK getf2:
=================================================================== */

/* ============================Setup
 * Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to
// templated testers; Some routines may not touch/use certain "members" of
// objects "argus". like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not
// have ldb, ldc; That is fine. These testers & routines will leave untouched
// members alone. Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like
// "std::get<0>" which is not intuitive and error-prone

Arguments setup_getf2_arguments(getf2_tuple tup) {

  vector<int> matrix_size = std::get<0>(tup);
  int n_size = std::get<1>(tup);

  Arguments arg;

  // see the comments about matrix_size_range above
  arg.M = matrix_size[0];
  arg.N = n_size;
  arg.lda = matrix_size[1];

  arg.timing = 0;

  return arg;
}

class getf2_gtest : public ::TestWithParam<getf2_tuple> {
protected:
  getf2_gtest() {}
  virtual ~getf2_gtest() {}
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_P(getf2_gtest, getf2_gtest_float) {
  // GetParam return a tuple. Tee setup routine unpack the tuple
  // and initializes arg(Arguments) which will be passed to testing routine
  // The Arguments data struture have physical meaning associated.
  // while the tuple is non-intuitive.

  Arguments arg = setup_getf2_arguments(GetParam());

  rocblas_status status = testing_getf2<float>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.M < 0 || arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

TEST_P(getf2_gtest, getf2_gtest_double) {
  // GetParam return a tuple. Tee setup routine unpack the tuple
  // and initializes arg(Arguments) which will be passed to testing routine
  // The Arguments data struture have physical meaning associated.
  // while the tuple is non-intuitive.

  Arguments arg = setup_getf2_arguments(GetParam());

  rocblas_status status = testing_getf2<double>(arg);

  // if not success, then the input argument is problematic, so detect the error
  // message
  if (status != rocblas_status_success) {

    if (arg.M < 0 || arg.N < 0) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    } else if (arg.lda < arg.M) {
      EXPECT_EQ(rocblas_status_invalid_size, status);
    }
  }
}

// notice we are using vector of vector
// so each element in xxx_range is a vector,
// ValuesIn take each element (a vector) and combine them and feed them to
// test_p The combinations are  { {M, N, lda}}

// This function mainly test the scope of matrix_size.
INSTANTIATE_TEST_CASE_P(daily_lapack, getf2_gtest,
                        Combine(ValuesIn(large_matrix_size_range),
                                ValuesIn(large_n_size_range)));

// THis function mainly test the scope of uplo_range, the scope of
// matrix_size_range is small
INSTANTIATE_TEST_CASE_P(checkin_lapack, getf2_gtest,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(n_size_range)));
