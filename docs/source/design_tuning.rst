.. _tuning_label:

*******************************
Tuning rocSOLVER Performance
*******************************

Some constant parameters in rocSOLVER can be modified to tune the performance of the
library functions in a given context (e.g. for a particular matrix size and/or shape).
A description of these tunable constants is presented in this section.

To facilitate the description, the constants are grouped by the function or family of functions they affect.
Some aspects of the involved algorithms are also depicted here for the sake of clarity; however,
this section is not intended to be a review of the well-known methods for different matrix computations.
These constants are specific to the rocSOLVER implementation of the algorithms, and are only
described within this context.

All described constants can be found in the header file
``<rocsolverDIR>/library/src/include/ideal_sizes.hpp``. Note that these are not run-time arguments for the
associated API functions. For any change to take effect, the library must be :ref:`rebuilt from source <userguide_install>`.

.. warning::
    The effect that changing the values of the tunable constants
    will have on the performance of the library is hard to predict, and this analysis is out
    of the scope of this document. Advanced users and developers trying to tune these values
    should proceed with caution; there is no guarantee that new values will improve (or worsen)
    the performance of the associated functions.

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top



geqr2/geqrf and geql2/geqlf functions
======================================

The orthogonal factorizations from the left (QR or QL factorizations) are separated into two versions,
blocked and unblocked. The unblocked routines GEQR2 and GEQL2 are based on BLAS level II operations and work by applying
Householder reflectors one column at a time. The blocked routines GEQRF and GEQLF factorize a block of columns at each
step using the unblocked functions (provided the matrix is large enough), and apply the resulting block reflectors to update
the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

GEQxF_BLOCKSIZE
----------------------
.. doxygendefine:: GEQxF_BLOCKSIZE

GEQxF_GEQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEQxF_GEQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



gerq2/gerqf and gelq2/gelqf functions
========================================

The orthogonal factorizations from the right (RQ or LQ factorizations) are separated into two versions,
blocked and unblocked. The unblocked routines GERQ2 and GELQ2 are based on BLAS level II operations and work by applying
Householder reflectors one row at a time. The blocked routines GERQF and GELQF factorize a block of rows at each
step using the unblocked functions (provided the matrix is large enough), and apply the resulting block reflectors to update
the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

GExQF_BLOCKSIZE
----------------------
.. doxygendefine:: GExQF_BLOCKSIZE

GExQF_GExQ2_SWITCHSIZE
-----------------------
.. doxygendefine:: GExQF_GExQ2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql functions
================================================================

The generators of a matrix Q with orthonormal columns (as products of Householder reflectors derived
from the QR or QL factorizations) are also separated into blocked and unblocked versions. The unblocked
routines ORG2R/UNG2R and ORG2L/UNG2L, based on BLAS level II operations, work by accumulating one Householder reflector at a time.
The blocked routines ORGQR/UNGQR and ORGQL/UNGQL multiply a set of reflectors at each step using the unblocked
functions (provided there are enough reflectors to accumulate), and apply the resulting block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

xxGQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxGQx_BLOCKSIZE

xxGQx_xxGQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxGQx_xxGQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq functions
================================================================

The generators of a matrix Q with orthonormal rows (as products of Householder reflectors derived
from the RQ or LQ factorizations) are also separated into blocked and unblocked versions. The unblocked
routines ORGR2/UNGR2 and ORGL2/UNGL2, based on BLAS level II operations, work by accumulating one Householder reflector at a time.
The blocked routines ORGRQ/UNGRQ and ORGLQ/UNGLQ multiply a set of reflectors at each step using the unblocked
functions (provided there are enough reflectors to accumulate), and apply the resulting block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

xxGxQ_BLOCKSIZE
----------------------
.. doxygendefine:: xxGxQ_BLOCKSIZE

xxGxQ_xxGxQ2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxGxQ_xxGxQ2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



orm2r/ormqr, orm2l/ormql, unm2r/unmqr and unm2l/unmql functions
================================================================

As with the generators of orthonormal/unitary matrices, the routines to multiply a general
matrix C by a matrix Q with orthonormal columns are separated into blocked and unblocked versions.
The unblocked routines ORM2R/UNM2R and ORM2L/UNM2L, based on BLAS level II operations, work by multiplying one Householder
reflector at a time, while the blocked routines ORMQR/UNMQR and ORMQL/UNMQL apply a set of reflectors at each step
(provided there are enough reflectors to start with).
The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

xxMQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxMQx_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq functions
================================================================

As with the generators of orthonormal/unitary matrices, the routines to multiply a general
matrix C by a matrix Q with orthonormal rows are separated into blocked and unblocked versions.
The unblocked routines ORMR2/UNMR2 and ORML2/UNML2, based on BLAS level II operations, work by multiplying one Householder
reflector at a time, while the blocked routines ORMRQ/UNMRQ and ORMLQ/UNMLQ apply a set of reflectors at each step
(provided there are enough reflectors to start with).
The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

xxMxQ_BLOCKSIZE
----------------------
.. doxygendefine:: xxMxQ_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



gebd2/gebrd and labrd functions
=================================

The computation of the bidiagonal form of a matrix is separated into blocked and
unblocked versions. The unblocked routine GEBD2 (and the auxiliary LABRD), based on BLAS level II operations,
apply Householder reflections to one column and row at a time. The blocked routine GEBRD reduces a leading block of rows and
columns at each step using the unblocked function LABRD (provided the matrix is large enough), and applies the resulting block reflectors to
update the trailing submatrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

GEBRD_BLOCKSIZE
---------------------
.. doxygendefine:: GEBRD_BLOCKSIZE

GEBRD_GEBD2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEBRD_GEBD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



gesvd function
==================

The Singular Value Decomposition of a matrix A could be sped up for matrices with sufficiently many more rows than
columns (or columns than rows) by starting with a QR factorization (or LQ factorization) of A, and working with the
triangular factor afterwards.

THIN_SVD_SWITCH
------------------
.. doxygendefine:: THIN_SVD_SWITCH

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



sytd2/sytrd, hetd2/hetrd and latrd functions
==============================================

The computation of the tridiagonal form of a symmetric/hermitian matrix is separated into blocked and
unblocked versions. The unblocked routines SYTD2/HETD2 (and the auxiliary LATRD), based on BLAS level II operations,
apply Householder reflections to one column/row at a time. The blocked routine SYTRD reduces a block of rows and columns at
each step using the unblocked function LATRD (provided the matrix is large enough), and applies the resulting block reflector to
update the rest of the matrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III), which,
in general, can give better performance on the GPU.

xxTRD_BLOCKSIZE
----------------------
.. doxygendefine:: xxTRD_BLOCKSIZE

xxTRD_xxTD2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxTRD_xxTD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



sygs2/sygst and hegs2/hegst functions
======================================

The reduction of a symmetric/hermitian-definite generalized eigenproblem to standard form is separated into
blocked and unblocked versions. The unblocked routines SYGS2/HEGS2 reduce the matrix A
one column/row at a time with vector operations and rank-2 updates (BLAS level II). The blocked
routines SYGST/HEGST reduce a leading block of A at each step using the unblocked methods (provided A is large enough),
and update the trailing matrix with BLAS level III operations (matrix products
and rank-2k updates), which, in general, can give better performance on the GPU.

xxGST_BLOCKSIZE
------------------------
.. doxygendefine:: xxGST_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



syevd, heevd and stedc functions
====================================

When running SYEVD/HEEVD, the computation of the eigenvectors of the associated tridiagonal matrix
can be sped up using a divide-and-conquer
approach (implemented in STEDC), provided the size of the independent block is large enough.

STEDC_MIN_DC_SIZE
-------------------
.. doxygendefine:: STEDC_MIN_DC_SIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



potf2/potrf functions
=========================

The Cholesky factorization is separated into blocked (right-looking) and unblocked versions. The unblocked
routine POTF2, based on BLAS level II operations, computes one diagonal element at a time,
and scales the corresponding row/column. The blocked routine POTRF factorizes a leading block of rows/columns
at each step using the unblocked algorithm (provided the matrix is large enough), and updates the trailing matrix with BLAS level III
operations (symmetric rank-k updates), which, in general, can give better performance on the GPU.

POTRF_BLOCKSIZE
------------------------
.. doxygendefine:: POTRF_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



sytf2/sytrf and lasyf functions
=================================

The Bunch-Kaufman factorization is separated into blocked and unblocked versions. The unblocked routine SYTF2
generates one 1-by-1 or 2-by-2 diagonal block at a time and applies a rank-1 update. The blocked routine SYTRF executes
a partial factorization of a given maximum number of diagonal elements (LASYF) at each step (provided the matrix is large enough),
and updates the rest of the matrix with matrix-matrix operations (BLAS level III), which, in general, can give better performance on the GPU.

SYTRF_BLOCKSIZE
----------------
.. doxygendefine:: SYTRF_BLOCKSIZE

SYTRF_SYTF2_SWITCHSIZE
-----------------------
.. doxygendefine:: SYTRF_SYTF2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).












getf2/getrf functions
========================

GETF2_MAX_COLS
---------------

GETF2_MAX_THDS
---------------

GETF2_OPTIM_NGRP
-----------------

GETRF_NUM_INTERVALS
--------------------

GETRF_INTERVALS
----------------

GETRF_BLKSIZES
---------------

GETRF_BATCH_NUM_INTERVALS
----------------------------

GETRF_BATCH_INTERVALS
----------------------

GETRF_BATCH_BLKSIZES
-------------------------

GETRF_NPVT_NUM_INTERVALS
--------------------------

GETRF_NPVT_INTERVALS
----------------------

GETRF_NPVT_BLKSIZES
---------------------

GETRF_NPVT_BATCH_NUM_INTERVALS
-------------------------------

GETRF_NPVT_BATCH_INTERVALS
---------------------------

GETRF_NPVT_BATCH_BLKSIZES
---------------------------




getri function
================

GETRI_MAX_COLS
---------------

GETRI_TINY_SIZE
----------------

GETRI_NUM_INTERVALS
--------------------

GETRI_INTERVALS
----------------

GETRI_BLKSIZES
----------------

GETRI_BATCH_TINY_SIZE
-----------------------

GETRI_BATCH_NUM_INTERVALS
--------------------------

GETRI_BATCH_INTERVALS
------------------------

GETRI_BATCH_BLKSIZES
---------------------


trtri function
=================

TRTRI_MAX_COLS
---------------

TRTRI_NUM_INTERVALS
--------------------

TRTRI_INTERVALS
----------------

TRTRI_BLKSIZES
---------------

TRTRI_BATCH_NUM_INTERVALS
--------------------------

TRTRI_BATCH_INTERVALS
----------------------

TRTRI_BATCH_BLKSIZES
---------------------











