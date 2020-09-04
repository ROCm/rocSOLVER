/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ARGUMENTS_H_
#define _ARGUMENTS_H_

#include "rocblas.h"

class Arguments {
public:
  rocblas_int M = 128;
  rocblas_int N = 128;
  rocblas_int K = 128;
  rocblas_int S4 = 128;
  rocblas_int k1 = 1;
  rocblas_int k2 = 2;

  rocblas_int lda = 128;
  rocblas_int ldb = 128;
  rocblas_int ldc = 128;
  rocblas_int ldv = 128;
  rocblas_int ldt = 128;

  rocblas_int incx = 1;
  rocblas_int incy = 1;
  rocblas_int incd = 1;
  rocblas_int incb = 1;

  rocblas_int start = 1024;
  rocblas_int end = 10240;
  rocblas_int step = 1000;

  double alpha = 1.0;
  double beta = 0.0;

  char transA_option = 'N';
  char transB_option = 'N';
  char transH_option = 'N';
  char side_option = 'L';
  char uplo_option = 'L';
  char diag_option = 'N';
  char direct_option = 'F';
  char storev = 'C';
  char left_svect = 'N';
  char right_svect = 'N';

  rocblas_int apiCallCount = 1;
  rocblas_int batch_count = 5;

  rocblas_int bsa =
      128 * 128; //  bsa > transA_option == 'N' ? lda * K : lda * M
  rocblas_int bsb =
      128 * 128; //  bsb > transB_option == 'N' ? ldb * N : ldb * K
  rocblas_int bsc = 128 * 128; //  bsc >= ldc * N
  rocblas_int bsp = 128;       //  bsp >= min(M,N)
  rocblas_int bs5 = 128;

  rocblas_int norm_check = 0;
  rocblas_int unit_check = 1;
  rocblas_int timing = 0;
  rocblas_int perf = 0;

  rocblas_int iters = 5;

  bool fast_alg = true;

  Arguments &operator=(const Arguments &rhs) {
    M = rhs.M;
    N = rhs.N;
    K = rhs.K;
    S4 = rhs.S4;
    k1 = rhs.k1;
    k2 = rhs.k2;

    lda = rhs.lda;
    ldb = rhs.ldb;
    ldc = rhs.ldc;
    ldv = rhs.ldv;
    ldt = rhs.ldt;

    incx = rhs.incx;
    incy = rhs.incy;
    incd = rhs.incd;
    incb = rhs.incb;

    start = rhs.start;
    end = rhs.end;
    step = rhs.step;

    alpha = rhs.alpha;
    beta = rhs.beta;

    transA_option = rhs.transA_option;
    transB_option = rhs.transB_option;
    transH_option = rhs.transH_option;
    side_option = rhs.side_option;
    uplo_option = rhs.uplo_option;
    diag_option = rhs.diag_option;
    direct_option = rhs.direct_option;
    storev = rhs.storev;
    left_svect = rhs.left_svect;
    right_svect = rhs.right_svect;

    apiCallCount = rhs.apiCallCount;
    batch_count = rhs.batch_count;

    bsa = rhs.bsa;
    bsb = rhs.bsb;
    bsc = rhs.bsc;
    bsp = rhs.bsp;
    bs5 = rhs.bs5;

    norm_check = rhs.norm_check;
    unit_check = rhs.unit_check;
    timing = rhs.timing;
    perf = rhs.perf;

    iters = rhs.iters;

    fast_alg = rhs.fast_alg;

    return *this;
  }
};

#endif
