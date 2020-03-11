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

#define GETRF_GETF2_SWITCHSIZE 64
#define POTRF_POTF2_SWITCHSIZE 64
#define GEQRF_GEQR2_SWITCHSIZE 128
#define GEQRF_GEQR2_BLOCKSIZE 64


#endif /* IDEAL_SIZES_HPP */
