/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _ARGUMENTS_H_
#define _ARGUMENTS_H_

#include "rocblas.h"

/* (TODO: The default values most be reviewed. Some combinations don't actually
    work for most of the tests) */

class Arguments
{
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

    rocblas_int bsa = 128 * 128;
    rocblas_int bsb = 128 * 128;
    rocblas_int bsc = 128 * 128;
    rocblas_int bsp = 128;
    rocblas_int bs5 = 128;

    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int timing = 0;
    rocblas_int perf = 0;
    rocblas_int singular = 0;

    rocblas_int iters = 5;
    char workmode = 'O';
};

#endif
