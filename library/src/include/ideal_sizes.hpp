/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

/*! \file
    \brief ideal_sizes.hpp gathers all constants that can be tuned for performance.
 *********************************************************************************/



/***************** geqr2/geqrf and geql2/geqlf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block column factorized at each step
    in the blocked QR or QL algorithm (GEQRF or GEQLF). */
#define GEQxF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing GEQRF or GEQLF.

    \details GEQRF or GEQLF will factorize blocks of GEQxF_BLOCKSIZE columns at a time until
    the trailing submatrix has no more than GEQxF_GEQx2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GEQR2 or GEQL2).*/
#define GEQxF_GEQx2_SWITCHSIZE 128



/***************** gerq2/gerqf and gelq2/gelqf ********************************
*******************************************************************************/
/*! \brief Determines the size of the block row factorized at each step
    in the blocked RQ or LQ algorithm (GERQF or GELQF). */
#define GExQF_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing GERQF or GELQF.

    \details GERQF or GELQF will factorize blocks of GExQF_BLOCKSIZE rows at a time until
    the trailing submatrix has no more than GExQF_GExQ2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be factorized with the unblocked algorithm (GERQ2 or GELQ2).*/
#define GExQF_GExQ2_SWITCHSIZE 128



/******** org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal columns with the blocked algorithm (ORGQR/UNGQR or ORGQL/UNGQL). */
#define xxGQx_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing ORGQR/UNGQR or ORGQL/UNGQL.

    \details ORGQR/UNGQR or ORGQL/UNGQL will accumulate xxGQx_BLOCKSIZE reflectors at a time until
    there is no more than xxGQx_xxGQx2_SWITCHSIZE reflectors left; the remaining reflectors, if any,
    are applied one by one with the unblocked algorithm (ORG2R/UNG2R or ORG2L/UNG2L).*/
#define xxGQx_xxGQx2_SWITCHSIZE 128



/******** orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that is applied at each step when
    generating a matrix Q with orthonormal rows with the blocked algorithm (ORGRQ/UNGRQ or ORGLQ/UNGLQ). */
#define xxGxQ_BLOCKSIZE 64

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing ORGRQ/UNGRQ or ORGLQ/UNGLQ.

    \details ORGRQ/UNGRQ or ORGLQ/UNGLQ will accumulate xxGxQ_BLOCKSIZE reflectors at a time until
    there is no more than xxGxQ_xxGxQ2_SWITCHSIZE reflectors left; the remaining  reflectors, if any,
    are applied one by one with the unblocked algorithm (ORGR2/UNGR2 or ORGL2/UNGL2).*/
#define xxGxQ_xxGxQ2_SWITCHSIZE 128



/********* orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql **************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMQR/UNMQR or ORMQL/UNMQL).

    \details If the total number of Householder reflectors is not multiple of xxMQx_BLOCKSIZE,
    the last block that updates C could have a smaller size.*/
#define xxMQx_BLOCKSIZE 64



/********* ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq ***************
*******************************************************************************/
/*! \brief Determines the size of the block reflector that multiplies the matrix C at each
    step with the blocked algorithm (ORMRQ/UNMRQ or ORMLQ/UNMLQ).

    \details If the total number of Householder reflectors is not multiple of xxMxQ_BLOCKSIZE,
    the last block that updates C could have a smaller size.*/
#define xxMxQ_BLOCKSIZE 64



/**************************** gebd2/gebrd *************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to bidiagonal form at each step
    when using the blocked algorithm (GEBRD). */
#define GEBRD_BLOCKSIZE 32

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing GEBRD.

    \details GEBRD will use LABRD to reduce blocks of GEBRD_BLOCKSIZE rows and columns at a time until
    the trailing submatrix has no more than GEBRD_GEBD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (GEBD2).*/
#define GEBRD_GEBD2_SWITCHSIZE 32



/******************************* gesvd ****************************************
*******************************************************************************/
/*! \brief Determines the factor by which one dimension of a matrix should exceed
    the other dimension for the thin SVD to be computed when executing GESVD.

    \details When a m-by-n matrix A is passed to GESVD, if m >= THIN_SVD_SWITCH*n or
    n >= THIN_SVD_SWITCH*m, then the thin SVD is computed.*/
#define THIN_SVD_SWITCH 1.6



/******************* sytd2/sytrd and hetd2/hetrd *******************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is reduced to tridiagonal form at each step
    when using the blocked algorithm (SYTRD/HETRD). */
#define xxTRD_BLOCKSIZE 32

/*! \brief Determines the size at which rocSOLVER switchs from
    the unblocked to the blocked algorithm when executing SYTRD/HETRD.

    \details SYTRD/HETRD will use LATRD to reduce blocks of xxTRD_BLOCKSIZE rows and columns at a time until
    the trailing submatrix has no more than xxTRD_xxTD2_SWITCHSIZE rows or columns; at this point the last block,
    if any, will be reduced with the unblocked algorithm (SYTD2/HETD2).*/
#define xxTRD_xxTD2_SWITCHSIZE 64



/***************** sygs2/sygst and hegs2/hegst ********************************
*******************************************************************************/
#define xxGST_BLOCKSIZE 64



/****************************** stedc *****************************************
*******************************************************************************/
#define STEDC_MIN_DC_SIZE 32



/************************** potf2/potrf ***************************************
*******************************************************************************/
#define POTRF_POTF2_SWITCHSIZE 64



/*************************** sytf2/sytrf **************************************
*******************************************************************************/
#define SYTRF_BLOCKSIZE 256



/**************************** getf2/getfr *************************************
*******************************************************************************/
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

