/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

/*! \file
    \brief ideal_sizes.hpp gathers all constants that can be tuned for performance.
 *********************************************************************************/

// org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql
#define xxGQx_xxGQx2_SWITCHSIZE 128
#define xxGQx_BLOCKSIZE 64

// orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq
#define xxGxQ_xxGxQ2_SWITCHSIZE 128
#define xxGxQ_BLOCKSIZE 64

// orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql
#define xxMQx_BLOCKSIZE 64

// ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq
#define xxMxQ_BLOCKSIZE 64

//// orgxx/ungxx
//#define ORGxx_UNGxx_SWITCHSIZE 128
//#define ORGxx_UNGxx_BLOCKSIZE 64

//// ormxx/unmxx
//#define ORMxx_ORMxx_BLOCKSIZE 32

// getf2/getfr
#define GETF2_MAX_COLS 64 //always <= wavefront size
#define GETF2_MAX_THDS 64
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#define GETRF_NUM_INTERVALS 4
#define GETRF_INTERVALS 64, 512, 1536, 4096
#define GETRF_BLKSIZES 0, 1, 32, 128, 384
#define GETRF_BATCH_NUM_INTERVALS 3
#define GETRF_BATCH_INTERVALS 52, 148, 1376
#define GETRF_BATCH_BLKSIZES 0, 16, 32, 288
#define GETRF_NPVT_NUM_INTERVALS 2
#define GETRF_NPVT_INTERVALS 65, 1536
#define GETRF_NPVT_BLKSIZES 0, 32, 256
#define GETRF_NPVT_BATCH_NUM_INTERVALS 3
#define GETRF_NPVT_BATCH_INTERVALS 33, 148, 1216
#define GETRF_NPVT_BATCH_BLKSIZES 0, 16, 32, 256

// getri
#define GETRI_MAX_COLS 64 //always <= wavefront size
#define GETRI_TINY_SIZE 43
#define GETRI_NUM_INTERVALS 1
#define GETRI_INTERVALS 1185
#define GETRI_BLKSIZES 0, 256
#define GETRI_BATCH_TINY_SIZE 35
#define GETRI_BATCH_NUM_INTERVALS 2
#define GETRI_BATCH_INTERVALS 505, 2049
#define GETRI_BATCH_BLKSIZES 32, 0, 256

// trtri
#define TRTRI_MAX_COLS 64 //always <= wavefront size
#define TRTRI_NUM_INTERVALS 1
#define TRTRI_INTERVALS 0
#define TRTRI_BLKSIZES 0, 0
#define TRTRI_BATCH_NUM_INTERVALS 3
#define TRTRI_BATCH_INTERVALS 32, 245, 1009
#define TRTRI_BATCH_BLKSIZES 0, 16, 32, 0

// potf2/potrf
#define POTRF_POTF2_SWITCHSIZE 64

// geqr2/geqrf and geql2/geqlf
#define GEQxF_GEQx2_SWITCHSIZE 128
#define GEQxF_BLOCKSIZE 64

// gerq2/gerqf and gelq2/gelqf
#define GExQF_GExQ2_SWITCHSIZE 128
#define GExQF_BLOCKSIZE 64

// gebd2/gebrd
#define GEBRD_GEBD2_SWITCHSIZE 32

// sytd2/sytrd and hetd2/hetrd
#define xxTRD_BLOCKSIZE 32
#define xxTRD_xxTD2_SWITCHSIZE 64

// sygs2/sygst and hegs2/hegst
#define xxGST_BLOCKSIZE 64

// gesvd
#define THIN_SVD_SWITCH 1.6

// stedc
#define STEDC_MIN_DC_SIZE 32

// sytf2/sytrf
#define SYTRF_BLOCKSIZE 256
