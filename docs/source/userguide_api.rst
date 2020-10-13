
*************
rocSOLVER API
*************

.. toctree::
   :maxdepth: 4
   :caption: Contents:

This section provides details of the rocSOLVER library API as in last ROCm release.

Types
=====

rocSOLVER Types
-----------------

Most rocSOLVER types are aliases of rocBLAS types.
See the `rocBLAS types <https://rocblas.readthedocs.io/en/latest/api.html#types>`_.

rocsolver_int
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_int
.. deprecated:: 3.5
   Use :c:type:`rocblas_int`.

rocsolver_handle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_handle
.. deprecated:: 3.5
   Use :c:type:`rocblas_handle`.

rocsolver_direction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_direction
.. deprecated:: 3.5
   Use :c:enum:`rocblas_direct`.

rocsolver_storev
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_storev
.. deprecated:: 3.5
   Use :c:enum:`rocblas_storev`.

rocsolver_operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_operation
.. deprecated:: 3.5
   Use :c:enum:`rocblas_operation`.

rocsolver_fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_fill
.. deprecated:: 3.5
   Use :c:enum:`rocblas_fill`.

rocsolver_diagonal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_diagonal
.. deprecated:: 3.5
   Use :c:enum:`rocblas_diagonal`.

rocsolver_side
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_side
.. deprecated:: 3.5
   Use :c:enum:`rocblas_side`.

rocsolver_status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_status
.. deprecated:: 3.5
   Use :c:enum:`rocblas_status`.


Additional Types
-------------------

rocblas_direct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_direct

rocblas_storev
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_storev

rocblas_svect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_svect

rocblas_workmode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_workmode


LAPACK Auxiliary Functions
============================

These are functions that support more advanced LAPACK routines.

Complex vector manipulations
--------------------------------------

rocsolver_<type>lacgv()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlacgv
   :outline:
.. doxygenfunction:: rocsolver_clacgv

Matrix permutations and manipulations
--------------------------------------

rocsolver_<type>laswp()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlaswp
   :outline:
.. doxygenfunction:: rocsolver_claswp
   :outline:
.. doxygenfunction:: rocsolver_dlaswp
   :outline:
.. doxygenfunction:: rocsolver_slaswp

Householder reflexions
--------------------------

rocsolver_<type>larfg()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlarfg
   :outline:
.. doxygenfunction:: rocsolver_clarfg
   :outline:
.. doxygenfunction:: rocsolver_dlarfg
   :outline:
.. doxygenfunction:: rocsolver_slarfg

rocsolver_<type>larft()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlarft
   :outline:
.. doxygenfunction:: rocsolver_clarft
   :outline:
.. doxygenfunction:: rocsolver_dlarft
   :outline:
.. doxygenfunction:: rocsolver_slarft

rocsolver_<type>larf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlarf
   :outline:
.. doxygenfunction:: rocsolver_clarf
   :outline:
.. doxygenfunction:: rocsolver_dlarf
   :outline:
.. doxygenfunction:: rocsolver_slarf

rocsolver_<type>larfb()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlarfb
   :outline:
.. doxygenfunction:: rocsolver_clarfb
   :outline:
.. doxygenfunction:: rocsolver_dlarfb
   :outline:
.. doxygenfunction:: rocsolver_slarfb

Bidiagonal forms
--------------------------

rocsolver_<type>labrd()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlabrd
   :outline:
.. doxygenfunction:: rocsolver_clabrd
   :outline:
.. doxygenfunction:: rocsolver_dlabrd
   :outline:
.. doxygenfunction:: rocsolver_slabrd

rocsolver_<type>bdsqr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zbdsqr
   :outline:
.. doxygenfunction:: rocsolver_cbdsqr
   :outline:
.. doxygenfunction:: rocsolver_dbdsqr
   :outline:
.. doxygenfunction:: rocsolver_sbdsqr


Orthonormal matrices
---------------------------

rocsolver_<type>org2r()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorg2r
   :outline:
.. doxygenfunction:: rocsolver_sorg2r

rocsolver_<type>orgqr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgqr
   :outline:
.. doxygenfunction:: rocsolver_sorgqr

rocsolver_<type>orgl2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgl2
   :outline:
.. doxygenfunction:: rocsolver_sorgl2

rocsolver_<type>orglq()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorglq
   :outline:
.. doxygenfunction:: rocsolver_sorglq

rocsolver_<type>org2l()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorg2l
   :outline:
.. doxygenfunction:: rocsolver_sorg2l

rocsolver_<type>orgql()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgql
   :outline:
.. doxygenfunction:: rocsolver_sorgql

rocsolver_<type>orgbr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgbr
   :outline:
.. doxygenfunction:: rocsolver_sorgbr

rocsolver_<type>orgtr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgtr
   :outline:
.. doxygenfunction:: rocsolver_sorgtr

rocsolver_<type>orm2r()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorm2r
   :outline:
.. doxygenfunction:: rocsolver_sorm2r

rocsolver_<type>ormqr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormqr
   :outline:
.. doxygenfunction:: rocsolver_sormqr

rocsolver_<type>orml2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorml2
   :outline:
.. doxygenfunction:: rocsolver_sorml2

rocsolver_<type>ormlq()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormlq
   :outline:
.. doxygenfunction:: rocsolver_sormlq

rocsolver_<type>orm2l()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorm2l
   :outline:
.. doxygenfunction:: rocsolver_sorm2l

rocsolver_<type>ormql()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormql
   :outline:
.. doxygenfunction:: rocsolver_sormql

rocsolver_<type>ormbr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormbr
   :outline:
.. doxygenfunction:: rocsolver_sormbr

rocsolver_<type>ormtr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormtr
   :outline:
.. doxygenfunction:: rocsolver_sormtr

Unitary matrices
---------------------------

rocsolver_<type>ung2r()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zung2r
   :outline:
.. doxygenfunction:: rocsolver_cung2r

rocsolver_<type>ungqr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zungqr
   :outline:
.. doxygenfunction:: rocsolver_cungqr

rocsolver_<type>ungl2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zungl2
   :outline:
.. doxygenfunction:: rocsolver_cungl2

rocsolver_<type>unglq()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunglq
   :outline:
.. doxygenfunction:: rocsolver_cunglq

rocsolver_<type>ung2l()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zung2l
   :outline:
.. doxygenfunction:: rocsolver_cung2l

rocsolver_<type>ungql()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zungql
   :outline:
.. doxygenfunction:: rocsolver_cungql

rocsolver_<type>ungbr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zungbr
   :outline:
.. doxygenfunction:: rocsolver_cungbr

rocsolver_<type>ungtr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zungtr
   :outline:
.. doxygenfunction:: rocsolver_cungtr

rocsolver_<type>unm2r()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunm2r
   :outline:
.. doxygenfunction:: rocsolver_cunm2r

rocsolver_<type>unmqr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunmqr
   :outline:
.. doxygenfunction:: rocsolver_cunmqr

rocsolver_<type>unml2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunml2
   :outline:
.. doxygenfunction:: rocsolver_cunml2

rocsolver_<type>unmlq()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunmlq
   :outline:
.. doxygenfunction:: rocsolver_cunmlq

rocsolver_<type>unm2l()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunm2l
   :outline:
.. doxygenfunction:: rocsolver_cunm2l

rocsolver_<type>unmql()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunmql
   :outline:
.. doxygenfunction:: rocsolver_cunmql

rocsolver_<type>unmbr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunmbr
   :outline:
.. doxygenfunction:: rocsolver_cunmbr

rocsolver_<type>unmtr()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zunmtr
   :outline:
.. doxygenfunction:: rocsolver_cunmtr



LAPACK Functions
==================

LAPACK routines solve complex Numerical Linear Algebra problems.

Special Matrix Factorizations
---------------------------------

rocsolver_<type>potf2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotf2
   :outline:
.. doxygenfunction:: rocsolver_cpotf2
   :outline:
.. doxygenfunction:: rocsolver_dpotf2
   :outline:
.. doxygenfunction:: rocsolver_spotf2

rocsolver_<type>potf2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_batched
   :outline:
.. doxygenfunction:: rocsolver_spotf2_batched

rocsolver_<type>potf2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotf2_strided_batched

rocsolver_<type>potrf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotrf
   :outline:
.. doxygenfunction:: rocsolver_cpotrf
   :outline:
.. doxygenfunction:: rocsolver_dpotrf
   :outline:
.. doxygenfunction:: rocsolver_spotrf

rocsolver_<type>potrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrf_batched

rocsolver_<type>potrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dpotrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_spotrf_strided_batched


General Matrix Factorizations
------------------------------

rocsolver_<type>getf2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2
   :outline:
.. doxygenfunction:: rocsolver_cgetf2
   :outline:
.. doxygenfunction:: rocsolver_dgetf2
   :outline:
.. doxygenfunction:: rocsolver_sgetf2

rocsolver_<type>getf2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_batched

rocsolver_<type>getf2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_strided_batched

rocsolver_<type>getrf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf
   :outline:
.. doxygenfunction:: rocsolver_cgetrf
   :outline:
.. doxygenfunction:: rocsolver_dgetrf
   :outline:
.. doxygenfunction:: rocsolver_sgetrf

rocsolver_<type>getrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_batched

rocsolver_<type>getrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_strided_batched

rocsolver_<type>geqr2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqr2
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2

rocsolver_<type>geqr2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_batched

rocsolver_<type>geqr2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqr2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqr2_strided_batched

.. _qr_label:

rocsolver_<type>geqrf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqrf
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf

.. _qr_batched_label:

rocsolver_<type>geqrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_batched

.. _qr_strided_label:

rocsolver_<type>geqrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqrf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqrf_strided_batched

rocsolver_<type>geql2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeql2
   :outline:
.. doxygenfunction:: rocsolver_cgeql2
   :outline:
.. doxygenfunction:: rocsolver_dgeql2
   :outline:
.. doxygenfunction:: rocsolver_sgeql2

rocsolver_<type>geql2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeql2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeql2_batched

rocsolver_<type>geql2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeql2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeql2_strided_batched

rocsolver_<type>geqlf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqlf
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf

rocsolver_<type>geqlf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf_batched

rocsolver_<type>geqlf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgeqlf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgeqlf_strided_batched

rocsolver_<type>gelq2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelq2
   :outline:
.. doxygenfunction:: rocsolver_cgelq2
   :outline:
.. doxygenfunction:: rocsolver_dgelq2
   :outline:
.. doxygenfunction:: rocsolver_sgelq2

rocsolver_<type>gelq2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelq2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelq2_batched

rocsolver_<type>gelq2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelq2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelq2_strided_batched

rocsolver_<type>gelqf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelqf
   :outline:
.. doxygenfunction:: rocsolver_cgelqf
   :outline:
.. doxygenfunction:: rocsolver_dgelqf
   :outline:
.. doxygenfunction:: rocsolver_sgelqf

rocsolver_<type>gelqf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelqf_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelqf_batched

rocsolver_<type>gelqf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgelqf_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgelqf_strided_batched

General Matrix Diagonalizations
-------------------------------

rocsolver_<type>gebd2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebd2
   :outline:
.. doxygenfunction:: rocsolver_cgebd2
   :outline:
.. doxygenfunction:: rocsolver_dgebd2
   :outline:
.. doxygenfunction:: rocsolver_sgebd2

rocsolver_<type>gebd2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebd2_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebd2_batched

rocsolver_<type>gebd2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebd2_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebd2_strided_batched

rocsolver_<type>gebrd()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebrd
   :outline:
.. doxygenfunction:: rocsolver_cgebrd
   :outline:
.. doxygenfunction:: rocsolver_dgebrd
   :outline:
.. doxygenfunction:: rocsolver_sgebrd

rocsolver_<type>gebrd_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebrd_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebrd_batched

rocsolver_<type>gebrd_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgebrd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgebrd_strided_batched

General Matrix Inversion
--------------------------

rocsolver_<type>getri()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetri
   :outline:
.. doxygenfunction:: rocsolver_cgetri
   :outline:
.. doxygenfunction:: rocsolver_dgetri
   :outline:
.. doxygenfunction:: rocsolver_sgetri

rocsolver_<type>getri_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_batched

rocsolver_<type>getri_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_strided_batched

General Systems Solvers
--------------------------

rocsolver_<type>getrs()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs
   :outline:
.. doxygenfunction:: rocsolver_cgetrs
   :outline:
.. doxygenfunction:: rocsolver_dgetrs
   :outline:
.. doxygenfunction:: rocsolver_sgetrs

rocsolver_<type>getrs_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_batched

rocsolver_<type>getrs_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrs_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrs_strided_batched


General Matrix Singular Value Decomposition
------------------------------------------------

rocsolver_<type>gesvd()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgesvd
   :outline:
.. doxygenfunction:: rocsolver_cgesvd
   :outline:
.. doxygenfunction:: rocsolver_dgesvd
   :outline:
.. doxygenfunction:: rocsolver_sgesvd

rocsolver_<type>gesvd_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvd_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvd_batched

rocsolver_<type>gesvd_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgesvd_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgesvd_strided_batched



Lapack-like Functions
========================

Other Lapack-like routines provided by rocSOLVER.

General Matrix Factorizations
---------------------------------

rocsolver_<type>getf2_npvt()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt

rocsolver_<type>getf2_npvt_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_batched

rocsolver_<type>getf2_npvt_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_strided_batched

rocsolver_<type>getrf_npvt()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt

rocsolver_<type>getrf_npvt_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_batched

rocsolver_<type>getrf_npvt_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_strided_batched



Auxiliaries
=========================

Auxiliary Functions
---------------------

rocSOLVER auxiliary functions are aliases of rocBLAS auxiliary functions.
See the `rocBLAS auxiliary functions <https://rocblas.readthedocs.io/en/latest/api.html#auxiliary>`_.

rocsolver_create_handle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_create_handle
.. deprecated:: 3.5
   Use :c:func:`rocblas_create_handle`.

rocsolver_destroy_handle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_destroy_handle
.. deprecated:: 3.5
   Use :c:func:`rocblas_destroy_handle`.

rocsolver_set_stream()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_stream
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_stream`.

rocsolver_get_stream()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_stream
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_stream`.

rocsolver_set_vector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_vector
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_vector`.

rocsolver_get_vector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_vector
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_vector`.

rocsolver_set_matrix()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_matrix
.. deprecated:: 3.5
   Use :c:func:`rocblas_set_matrix`.

rocsolver_get_matrix()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_matrix
.. deprecated:: 3.5
   Use :c:func:`rocblas_get_matrix`.
