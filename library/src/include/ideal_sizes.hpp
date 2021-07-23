/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

// general
#define WAVESIZE 64 // size of wavefront

// These are used by different common kernels
//(TODO: identify functions and name accordingly)
#define BLOCKSIZE 256
#define BS 32

// laswp
#define LASWP_BLOCKSIZE 256

// orgxx/ungxx
#define ORGxx_UNGxx_SWITCHSIZE 128
#define ORGxx_UNGxx_BLOCKSIZE 64

// ormxx/unmxx
#define ORMxx_ORMxx_BLOCKSIZE 32

// getf2/getfr
#define GETF2_MAX_THDS 256
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#define GETF2_BATCH_OPTIM_MAX_SIZE 1024
#define GETF2_OPTIM_MAX_SIZE 256
#define GETRF_NUM_INTERVALS 4
#define GETRF_INTERVALS 65, 673, 2017, 3264
#define GETRF_BLKSIZES 1, 32, 128, 384, 256
#define GETRF_BATCH_NUM_INTERVALS 3
#define GETRF_BATCH_INTERVALS 65, 497, 2049
#define GETRF_BATCH_BLKSIZES 1, 16, 32, 64
#define GETRF_NPVT_NUM_INTERVALS 3
#define GETRF_NPVT_INTERVALS 65, 3073, 4609
#define GETRF_NPVT_BLKSIZES 1, 32, 64, 192
#define GETRF_NPVT_BATCH_NUM_INTERVALS 3
#define GETRF_NPVT_BATCH_INTERVALS 45, 181, 2049
#define GETRF_NPVT_BATCH_BLKSIZES 1, 16, 32, 64

// getri
#define GETRI_TINY_SIZE 43
#define GETRI_NUM_INTERVALS 1
#define GETRI_INTERVALS 1185
#define GETRI_BLKSIZES 0, 256
#define GETRI_BATCH_TINY_SIZE 35
#define GETRI_BATCH_NUM_INTERVALS 2
#define GETRI_BATCH_INTERVALS 505, 2049
#define GETRI_BATCH_BLKSIZES 32, 0, 256

// TRTRI
#define TRTRI_NUM_INTERVALS 1
#define TRTRI_INTERVALS 0
#define TRTRI_BLKSIZES 0, 0
#define TRTRI_BATCH_NUM_INTERVALS 3
#define TRTRI_BATCH_INTERVALS 32, 245, 1009
#define TRTRI_BATCH_BLKSIZES 0, 16, 32, 0

// potf2/potrf
#define POTRF_POTF2_SWITCHSIZE 64

// geqx2/geqxf
#define GEQxF_GEQx2_SWITCHSIZE 128
#define GEQxF_GEQx2_BLOCKSIZE 64

// gexq2/gexqf
#define GExQF_GExQ2_SWITCHSIZE 128
#define GExQF_GExQ2_BLOCKSIZE 64

// gebd2/gebrd
#define GEBRD_GEBD2_SWITCHSIZE 32

// xxtd2/xxtrd
#define xxTRD_xxTD2_BLOCKSIZE 32
#define xxTRD_xxTD2_SWITCHSIZE 64

// xxgs2/xxgst
#define xxGST_xxGS2_BLOCKSIZE 64

// gesvd
#define THIN_SVD_SWITCH 1.6

// STEDC
#define STEDC_MIN_DC_SIZE 32

// THESE FOLLOWING VALUES ARE TO MATCH ROCBLAS C++ INTERFACE
// THEY ARE DEFINED/TUNNED IN ROCBLAS
#define ROCBLAS_AXPY_NB 256
#define ROCBLAS_SCAL_NB 256
#define ROCBLAS_DOT_NB 512
#define ROCBLAS_TRMV_NB 512
#define ROCBLAS_TRMM_REAL_NB 32
#define ROCBLAS_TRMM_COMPLEX_NB 16
#define ROCBLAS_IAMAX_NB 1024
#define ROCBLAS_TRSV_BLOCK 64
#define ROCBLAS_TRSV_Z_BLOCK 32
#define ROCBLAS_TRSM_BLOCK 128
#define ROCBLAS_TRTRI_NB 16
