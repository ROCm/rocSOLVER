/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef IDEAL_SIZES_HPP
#define IDEAL_SIZES_HPP

// general
#define WAVESIZE 64 //size of wavefront

// These are used by different common kernels
//(TODO: identify functions and name accordingly)
#define BLOCKSIZE 256
#define BS 32  

// laswp
#define LASWP_BLOCKSIZE 256

// ormqr/orm2r & ormlq_orml2
#define ORMQR_ORM2R_BLOCKSIZE 32
#define ORMLQ_ORML2_BLOCKSIZE 32

// getf2/getfr
#define GETRF_GETF2_SWITCHSIZE 64
#define GETF2_MAX_THDS 256
#define GETRF_GETF2_BLOCKSIZE 64
#define GETF2_OPTIM_NGRP 16,15,8,8,8,8,8,8,6,6,4,4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2
#define GETF2_BATCH_OPTIM_MAX_SIZE 2048
#define GETF2_OPTIM_MAX_SIZE 1024

// getri
#define GETRI_SWITCHSIZE_MID 64
#define GETRI_SWITCHSIZE_LARGE 64
#define GETRI_BLOCKSIZE 64
#define TRTRI_SWITCHSIZE_MID 64
#define TRTRI_SWITCHSIZE_LARGE 512
#define TRTRI_BLOCKSIZE 64

// potf2/potrf
#define POTRF_POTF2_SWITCHSIZE 64

// geqr2/geqrf
#define GEQRF_GEQR2_SWITCHSIZE 128
#define GEQRF_GEQR2_BLOCKSIZE 64

// gebd2/gebrd
#define GEBRD_GEBD2_SWITCHSIZE 32

// THESE FOLLOWING VALUES ARE TO MATCH ROCBLAS C++ INTERFACE
// THEY ARE DEFINED/TUNNED IN ROCBLAS
#define ROCBLAS_SCAL_NB 256
#define ROCBLAS_DOT_NB 512
#define ROCBLAS_TRMV_NB 512
#define ROCBLAS_TRMM_NB 128
#define ROCBLAS_IAMAX_NB 1024
#define ROCBLAS_TRSM_BLOCK 128


#endif /* IDEAL_SIZES_HPP */
