/* **************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

/*! \file
    \brief ideal_sizes.hpp gathers all constants that can be tuned for performance.
 *********************************************************************************/

/***************** geqr2/geqrf and geql2/geqlf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block column factorized at each step
    in the blocked QR or QL algorithm (GEQRF or GEQLF). It also applies to the
    corresponding batched and strided-batched routines. */
#define GEQxF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GEQRF or GEQLF. It also applies to the
    corresponding batched and strided-batched routines.

    \details GEQRF or GEQLF will factorize blocks of GEQxF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than GEQxF_GEQx2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GEQR2 or GEQL2).*/
#define GEQxF_GEQx2_SWITCHSIZE 128

/***************** gerq2/gerqf and gelq2/gelqf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block row factorized at each step
    in the blocked RQ or LQ algorithm (GERQF or GELQF). It also applies to the
    corresponding batched and strided-batched routines. */
#define GExQF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GERQF or GELQF. It also applies to the
    corresponding batched and strided-batched routines.

    \details GERQF or GELQF will factorize blocks of GExQF_BLOCKSIZE rows at a time until
    the rest of the matrix has no more than GExQF_GExQ2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GERQ2 or GELQ2).*/
#define GExQF_GExQ2_SWITCHSIZE 128

/******** org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal columns with the blocked algorithm (ORGQR/UNGQR or ORGQL/UNGQL). */
#define xxGQx_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing ORGQR/UNGQR or ORGQL/UNGQL.

    \details ORGQR/UNGQR or ORGQL/UNGQL will accumulate xxGQx_BLOCKSIZE reflectors at a time until
    there are no more than xxGQx_xxGQx2_SWITCHSIZE reflectors left; the remaining reflectors, if any,
    are applied one by one using the unblocked algorithm (ORG2R/UNG2R or ORG2L/UNG2L).*/
#define xxGQx_xxGQx2_SWITCHSIZE 128

/******** orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal rows with the blocked algorithm (ORGRQ/UNGRQ or ORGLQ/UNGLQ). */
#define xxGxQ_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing ORGRQ/UNGRQ or ORGLQ/UNGLQ.

    \details ORGRQ/UNGRQ or ORGLQ/UNGLQ will accumulate xxGxQ_BLOCKSIZE reflectors at a time until
    there are no more than xxGxQ_xxGxQ2_SWITCHSIZE reflectors left; the remaining reflectors, if any,
    are applied one by one using the unblocked algorithm (ORGR2/UNGR2 or ORGL2/UNGL2).*/
#define xxGxQ_xxGxQ2_SWITCHSIZE 128

/********* orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMQR/UNMQR or ORMQL/UNMQL).

    \details xxMQx_BLOCKSIZE also acts as a switch size; if the total number of reflectors is not greater than xxMQx_BLOCKSIZE (k <= xxMQx_BLOCKSIZE),
    ORMQR/UNMQR or ORMQL/UNMQL will directly call the unblocked routines (ORM2R/UNM2R or ORM2L/UNM2L). However, when k is not a multiple of xxMQx_BLOCKSIZE,
    the last block that updates C in the blocked process is allowed to be smaller than xxMQx_BLOCKSIZE.*/
#define xxMQx_BLOCKSIZE 64

/********* ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMRQ/UNMRQ or ORMLQ/UNMLQ).

    \details xxMxQ_BLOCKSIZE also acts as a switch size; if the total number of reflectors is not greater than xxMxQ_BLOCKSIZE (k <= xxMxQ_BLOCKSIZE),
    ORMRQ/UNMRQ or ORMLQ/UNMLQ will directly call the unblocked routines (ORMR2/UNMR2 or ORML2/UNML2). However, when k is not a multiple of xxMxQ_BLOCKSIZE,
    the last block that updates C in the blocked process is allowed to be smaller than xxMxQ_BLOCKSIZE.*/
#define xxMxQ_BLOCKSIZE 64

/**************************** gebd2/gebrd *************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to bidiagonal form at each step
    when using the blocked algorithm (GEBRD). It also applies to the
    corresponding batched and strided-batched routines.*/
#define GEBRD_BLOCKSIZE 32

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing GEBRD. It also applies to the
    corresponding batched and strided-batched routines.

    \details GEBRD will use LABRD to reduce blocks of GEBRD_BLOCKSIZE rows and columns at a time until
    the trailing submatrix has no more than GEBRD_GEBD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (GEBD2).*/
#define GEBRD_GEBD2_SWITCHSIZE 64

/******************************* bdsqr ****************************************
*******************************************************************************/
/*! \brief Determines the maximum number of split diagonal blocks that BDSQR can process in parallel.
    Must be at least 1.

    \details BDSQR will use BDSQR_SPLIT_GROUPS thread groups in order to process diagonal blocks
    in parallel. */
#define BDSQR_SPLIT_GROUPS 5

/******************************* gesvd ****************************************
*******************************************************************************/
/*! \brief Determines the factor by which one dimension of a matrix should exceed
    the other dimension for the thin SVD to be computed when executing GESVD. It also applies to the
    corresponding batched and strided-batched routines.

    \details When a m-by-n matrix A is passed to GESVD, if m >= THIN_SVD_SWITCH*n or
    n >= THIN_SVD_SWITCH*m, then the thin SVD is computed.*/
#define THIN_SVD_SWITCH 1.6

/******************* sytd2/sytrd and hetd2/hetrd *******************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to tridiagonal form at each step
    when using the blocked algorithm (SYTRD/HETRD). It also applies to the
    corresponding batched and strided-batched routines.*/
#define xxTRD_BLOCKSIZE 32

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing SYTRD/HETRD. It also applies to the
    corresponding batched and strided-batched routines.

    \details SYTRD/HETRD will use LATRD to reduce blocks of xxTRD_BLOCKSIZE rows and columns at a time until
    the rest of the matrix has no more than xxTRD_xxTD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (SYTD2/HETD2).*/
#define xxTRD_xxTD2_SWITCHSIZE 64

/***************** sygs2/sygst and hegs2/hegst ********************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to standard form at each step
    when using the blocked algorithm (SYGST/HEGST). It also applies to the
    corresponding batched and strided-batched routines.

    \details xxGST_BLOCKSIZE also acts as a switch size; if the original size of the problem is not larger than xxGST_BLOCKSIZE (n <= xxGST_BLOCKSIZE),
    SYGST/HEGST will directly call the unblocked routines (SYGS2/HEGS2). However, when n is not a
    multiple of xxGST_BLOCKSIZE, the last block reduced in the blocked process is allowed to be smaller than xxGST_BLOCKSIZE.*/
#define xxGST_BLOCKSIZE 64

/****************************** stedc ******************************************
*******************************************************************************/
/*! \brief Determines the minimum size required for the eigenvectors of an independent block of
    a tridiagonal matrix to be computed using the divide-and-conquer algorithm (STEDC).

    \details If the size of the block is smaller than STEDC_MIN_DC_SIZE (bs < STEDC_MIN_DC_SIZE),
    the eigenvectors are computed with the normal QR algorithm. */
#define STEDC_MIN_DC_SIZE 16

/*! \brief Determines the number of split blocks (independent blocks) of a tridiagonal matrix that
    are analyzed in parallel with the divide & conquer method. */
#define STEDC_NUM_SPLIT_BLKS 8

/************************** potf2/potrf ***************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is factorized at each step
    when using the blocked algorithm (POTRF). It also applies to the
    corresponding batched and strided-batched routines.*/
#define POTRF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing POTRF. It also applies to the
    corresponding batched and strided-batched routines.

    \details POTRF will factorize blocks of POTRF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than POTRF_POTF2_SWITCHSIZE columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (POTF2).*/
#define POTRF_POTF2_SWITCHSIZE 128

/************************** syevj/heevj ***************************************
*******************************************************************************/
/*! \brief Determines the size at which rocSOLVER switches from
    the small-size kernel to the blocked algorithm when executing SYEVJ. It also applies to the
    corresponding batched and strided-batched routines. Must be <= 64.

    \details If the size of the matrix is not greater than SYEVJ_BLOCKED_SWITCH, the eigenvalues
    and eigenvectors will be computed with a single kernel call. */
#define SYEVJ_BLOCKED_SWITCH 58

/*************************** sytf2/sytrf **************************************
*******************************************************************************/
/*! \brief Determines the maximum size of the partial factorization executed at each step
    when using the blocked algorithm (SYTRF). It also applies to the
    corresponding batched and strided-batched routines.*/
#define SYTRF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing SYTRF. It also applies to the
    corresponding batched and strided-batched routines.

    \details SYTRF will use LASYF to factorize a submatrix of at most SYTRF_BLOCKSIZE columns at a time until
    the rest of the matrix has no more than SYTRF_SYTF2_SWITCHSIZE columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (SYTF2).*/
#define SYTRF_SYTF2_SWITCHSIZE 128

/**************************** getf2/getfr *************************************
*******************************************************************************/
#define GETF2_SPKER_MAX_M 1024 //always <= 1024
#define GETF2_SPKER_MAX_N 256 //always <= 256
#define GETF2_SSKER_MAX_M 512 //always <= 512 and <= GETF2_SPKER_MAX_M
#define GETF2_SSKER_MAX_N 64 //always <= wavefront and <= GETF2_SPKER_MAX_N
#define GETF2_OPTIM_NGRP \
    16, 15, 8, 8, 8, 8, 8, 8, 6, 6, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
#define GETRF_NUM_INTERVALS_REAL 4
#define GETRF_INTERVALS_REAL 64, 512, 1856, 2944
#define GETRF_BLKSIZES_REAL 0, 1, 32, 256, 512
#define GETRF_BATCH_NUM_INTERVALS_REAL 9
#define GETRF_BATCH_INTERVALS_REAL 40, 42, 46, 49, 52, 58, 112, 800, 1024
#define GETRF_BATCH_BLKSIZES_REAL 0, 32, 0, 16, 0, 32, 1, 32, 64, 160
#define GETRF_NPVT_NUM_INTERVALS_REAL 2
#define GETRF_NPVT_INTERVALS_REAL 64, 512
#define GETRF_NPVT_BLKSIZES_REAL 0, -1, 512
#define GETRF_NPVT_BATCH_NUM_INTERVALS_REAL 6
#define GETRF_NPVT_BATCH_INTERVALS_REAL 40, 168, 448, 512, 896, 1408
#define GETRF_NPVT_BATCH_BLKSIZES_REAL 0, -24, -32, -64, 32, 96, 512

#define GETRF_NUM_INTERVALS_COMPLEX 4
#define GETRF_INTERVALS_COMPLEX 64, 512, 1024, 2944
#define GETRF_BLKSIZES_COMPLEX 0, 1, 32, 96, 512
#define GETRF_BATCH_NUM_INTERVALS_COMPLEX 10
#define GETRF_BATCH_INTERVALS_COMPLEX 23, 28, 30, 32, 40, 48, 56, 64, 768, 1024
#define GETRF_BATCH_BLKSIZES_COMPLEX 0, 16, 0, 1, 24, 16, 24, 16, 48, 64, 160
#define GETRF_NPVT_NUM_INTERVALS_COMPLEX 2
#define GETRF_NPVT_INTERVALS_COMPLEX 64, 512
#define GETRF_NPVT_BLKSIZES_COMPLEX 0, -1, 512
#define GETRF_NPVT_BATCH_NUM_INTERVALS_COMPLEX 5
#define GETRF_NPVT_BATCH_INTERVALS_COMPLEX 20, 32, 42, 512, 1408
#define GETRF_NPVT_BATCH_BLKSIZES_COMPLEX 0, -16, -32, -48, 64, 128

/****************************** getri *****************************************
*******************************************************************************/
#define GETRI_MAX_COLS 64 //always <= wavefront size
#define GETRI_TINY_SIZE 43
#define GETRI_NUM_INTERVALS 1
#define GETRI_INTERVALS 1185
#define GETRI_BLKSIZES 0, 256
#define GETRI_BATCH_TINY_SIZE 35
#define GETRI_BATCH_NUM_INTERVALS 2
#define GETRI_BATCH_INTERVALS 505, 2049
#define GETRI_BATCH_BLKSIZES 32, 0, 256

/***************************** trtri ******************************************
*******************************************************************************/
#define TRTRI_MAX_COLS 64 //always <= wavefront size
#define TRTRI_NUM_INTERVALS 1
#define TRTRI_INTERVALS 0
#define TRTRI_BLKSIZES 0, 0
#define TRTRI_BATCH_NUM_INTERVALS 3
#define TRTRI_BATCH_INTERVALS 32, 245, 1009
#define TRTRI_BATCH_BLKSIZES 0, 16, 32, 0
