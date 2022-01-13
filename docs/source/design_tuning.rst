.. _tuning_label:

*******************************
Tuning rocSOLVER Performance
*******************************

Some constant parameters in rocSOLVER can be modified to tune the performance of the
library functions in a given context (like for a particular matrix size and/or shape, for example).
A description of these tunable constants is presented in this section.

To facilitate the description, the constants are grouped by the function or familly of functions they affect.
Some aspects of the involved algorithms are also depicted here for the sake of clarity, however,
this section if not intended to be a review of the well known methods for the different matrix computations.
These constants are specific to the rocSOLVER implementation of the algorithms, and are only
described within this context.

All the described constants are gathered in the header file
``<rocsolverDIR>/library/src/include/ideal_sizes.hpp``, and
for any change to take effect, the library must be :ref:`rebuilt from source <userguide_install>`;
these are not run-time arguments of the associated API functions.

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

The orthogonal factorizations from the left (QR or QL factorizations) are implemented in two routines:
blocked and unblocked. The unblocked routines (GEQR2 or GEQL2) are based on BLAS level II operations and work applying
Householder reflectors one column at a time. The blocked routines (GEQRF or GEQLF), providing the matrix is large enough,
factorize a block of columns at each step using the unblocked functions, and apply the resulting block reflectors to
update the trailing submatrices. The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

GEQxF_BLOCKSIZE
----------------------
.. doxygendefine:: GEQxF_BLOCKSIZE

GEQxF_GEQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEQxF_GEQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



gerq2/gerqf and gelq2/gelqf functions
========================================

The orthogonal factorizations from the right (RQ or LQ factorizations) are implemented in two routines:
blocked and unblocked. The unblocked routines (GERQ2 or GELQ2) are based on BLAS level II operations and work applying
Householder reflectors one row at a time. The blocked routines (GERQF or GELQF), providing the matrix is large enough,
factorize a block of rows at each step using the unblocked functions, and apply the resulting block reflectors to
update the trailing submatrices. The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

GExQF_BLOCKSIZE
----------------------
.. doxygendefine:: GExQF_BLOCKSIZE

GExQF_GExQ2_SWITCHSIZE
-----------------------
.. doxygendefine:: GExQF_GExQ2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



org2r/orgqr, org2l/orgql, ung2r/ungqr and ung2l/ungql functions
================================================================

The generators of a matrix Q with orthonormal columns, as products of Householder reflectors derived
from the QR or QL factorizations, are also implemented in blocked and unblocked versions. The unblocked
routines (ORG2R/UNG2R and ORG2L/UNG2L), based on BLAS level II operations, work by accumulating one Householder reflector at a time.
The blocked routines (ORGQR/UNGQR and ORGQL/UNGQL), providing there is enough reflectors to accumulate, multiply a set
of reflectors at each step using the unblocked functions, and apply the resulting block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

xxGQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxGQx_BLOCKSIZE

xxGQx_xxGQx2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxGQx_xxGQx2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



orgr2/orgrq, orgl2/orglq, ungr2/ungrq and ungl2/unglq functions
================================================================

The generators a the matrix Q with orthonormal rows, as products of Householder reflectors derived
from the RQ or LQ factorizations, are also implemented in blocked and unblocked versions. The unblocked
routines (ORGR2/UNGR2 and ORGL2/UNGL2), based on BLAS level II operations, work by accumulating one Householder reflector at a time.
The blocked routines (ORGRQ/UNGRQ and ORGLQ/UNGLQ), providing there is enough reflectors to accumulate, multiply a set
of reflectors at each step using the unblocked functions, and apply the temporary block reflector to update Q.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

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
matrix C by a matrix Q with orthonormal columns are implemented in blocked and unblocked versions.
The unblocked routines (ORM2R/UNM2R and ORM2L/UNM2L),
based on BLAS level II operations, work by multiplying one Householder reflector at a time, while the
blocked routines (ORMQR/UNMQR and ORMQL/UNMQL) apply a set of reflectors at each step.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

xxMQx_BLOCKSIZE
----------------------
.. doxygendefine:: xxMQx_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



ormr2/ormrq, orml2/ormlq, unmr2/unmrq and unml2/unmlq functions
================================================================

As with the generators orthonormal/unitary matrices, the routines to multiply a general
matrix C by a matrix Q with orthonormal rows are implemented in blocked and unblocked versions.
The unblocked routines (ORMR2/UNMR2 and ORML2/UNML2),
based on BLAS level II operations, work by multiplying one Householder reflector at a time, while the
blocked routines (ORMRQ/UNMRQ and ORMLQ/UNMLQ) apply a set of reflectors at each step.
The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

xxMxQ_BLOCKSIZE
----------------------
.. doxygendefine:: xxMxQ_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



gebd2/gebrd and labrd functions
=================================

The computation of the bidiagonal form of a matrix is implemented in blocked and
unblocked versions. The unblocked routines (GEBD2 and the auxiliary LABRD), based on BLAS level II operations,
apply Householder reflections to one column and row at a time. The blocked routine (GEBRD), providing the matrix is large enough,
reduces a block of rows and columns at each step using the unblocked function LABRD, and apply the resulting block reflectors to
update the trailing submatrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU.

GEBRD_BLOCKSIZE
---------------------
.. doxygendefine:: GEBRD_BLOCKSIZE

GEBRD_GEBD2_SWITCHSIZE
-----------------------
.. doxygendefine:: GEBRD_GEBD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



gesvd function
==================

The Singular Value Decomposition could be speed up for matrices with sufficiently more rows than
columns (or columns than rows) by starting with a QR factorization (or LQ
factorization) and working with the triangular factor afterwards.

THIN_SVD_SWITCH
------------------
.. doxygendefine:: THIN_SVD_SWITCH

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



sytd2/sytrd, hetd2/hetrd and latrd functions
==============================================

The computation of the tridiagonal form of a symmetric/hermitian matrix is implemented in blocked and
unblocked versions. The unblocked routines (SYTD2/HETD2 and the auxiliary LATRD), based on BLAS level II operations,
apply Householder reflections to one column/row at a time. The blocked routine (SYTRD), providing the matrix is large enough,
reduces a block of rows and columns at each step using the unblocked function LATRD, and apply the resulting block reflector to
update the trailing submatrix. The application of the block reflectors is based on matrix-matrix operations (BLAS level III) which,
in general, could have better performance on the GPU

xxTRD_BLOCKSIZE
----------------------
.. doxygendefine:: xxTRD_BLOCKSIZE

xxTRD_xxTD2_SWITCHSIZE
-----------------------
.. doxygendefine:: xxTRD_xxTD2_SWITCHSIZE

(As of the current rocSOLVER release, these constants have not been tuned for any particular case).



sygs2/sygst and hegs2/hegst functions
======================================

xxGST_BLOCKSIZE
------------------------
.. doxygendefine:: xxGST_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



stedc function
===================

STEDC_MIN_DC_SIZE
-------------------
.. doxygendefine:: STEDC_MIN_DC_SIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



potf2/potrf functions
=========================

POTRF_POTF2_SWITCHSIZE
------------------------
.. doxygendefine:: POTRF_POTF2_SWITCHSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).



sytf2/sytrf functions
=======================

SYTRF_BLOCKSIZE
----------------
.. doxygendefine:: SYTRF_BLOCKSIZE

(As of the current rocSOLVER release, this constant has not been tuned for any particular case).












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











