/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef IDEAL_SIZES_HPP
#define IDEAL_SIZES_HPP


// IDEAL SIZES ARE DEFINED FOR NOW AS IN CPU-LAPACK
// BENCHMARKING OF ROCSOLVER WILL BE NEEDED TO DETERMINE
// MORE SUITABLE VALUES  
#define BLOCKSIZE 256
#define LASWP_BLOCKSIZE 256
#define GETF2_BLOCKSIZE 256
#define ORMQR_ORM2R_BLOCKSIZE 32
#define ORMLQ_ORML2_BLOCKSIZE 32
#define GETRF_GETF2_SWITCHSIZE 64
#define POTRF_POTF2_SWITCHSIZE 64
#define GEQRF_GEQR2_SWITCHSIZE 128
#define GEQRF_GEQR2_BLOCKSIZE 64
#define GEBRD_GEBD2_SWITCHSIZE 64
#define BS 32 //blocksize for kernels

// THESE VALUES ARE TO MATCH ROCBLAS C++ INTERFACE
// THEY ARE DEFINED/TUNNED IN ROCBLAS
#define ROCBLAS_SCAL_NB 256
#define ROCBLAS_DOT_NB 512
#define ROCBLAS_TRMV_NB 512


#endif /* IDEAL_SIZES_HPP */
