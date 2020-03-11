.. toctree::
   :maxdepth: 4 
   :caption: Contents:

*************
rocSOLVER API
*************

This section provides details of the rocSOLVER library API as in release 
`ROCm 2.10 <https://github.com/ROCmSoftwarePlatform/rocSOLVER/tree/master-rocm-2.10>`_.



Types
=====

Most rocSOLVER types are aliases of rocBLAS types. 
See rocBLAS types `here <https://rocblas.readthedocs.io/en/latest/api.html#types>`_.

Definitions
----------------

rocsolver_int
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_int

Enums
------------

rocsolver_handle
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_handle

rocsolver_operation
^^^^^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_operation

rocsolver_fill
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_fill

rocsolver_diagonal
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_diagonal

rocsolver_side
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_side

rocsolver_direct
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocsolver_direct

rocsolver_storev
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocsolver_storev

rocsolver_status
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_status




Lapack Auxiliary Functions
============================

These are functions that support more advanced Lapack routines.

Matrix permutations and manipulations
--------------------------------------

rocsolver_<type>laswp()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zlaswp
.. doxygenfunction:: rocsolver_claswp
.. doxygenfunction:: rocsolver_dlaswp
.. doxygenfunction:: rocsolver_slaswp

Householder reflexions
--------------------------

rocsolver_<type>larfg()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarfg
.. doxygenfunction:: rocsolver_slarfg

rocsolver_<type>larft()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarft
.. doxygenfunction:: rocsolver_slarft

rocsolver_<type>larf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarf
.. doxygenfunction:: rocsolver_slarf

rocsolver_<type>larfb()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarfb
.. doxygenfunction:: rocsolver_slarfb

Orthonormal matrices
---------------------------

rocsolver_<type>org2r()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorg2r
.. doxygenfunction:: rocsolver_sorg2r

rocsolver_<type>orgqr()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgqr
.. doxygenfunction:: rocsolver_sorgqr

rocsolver_<type>orgl2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgl2
.. doxygenfunction:: rocsolver_sorgl2

rocsolver_<type>orglq()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorglq
.. doxygenfunction:: rocsolver_sorglq

rocsolver_<type>orgbr()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorgbr
.. doxygenfunction:: rocsolver_sorgbr

rocsolver_<type>orm2r()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dorm2r
.. doxygenfunction:: rocsolver_sorm2r

rocsolver_<type>ormqr()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dormqr
.. doxygenfunction:: rocsolver_sormqr


Lapack Functions
==================

Lapack routines solve complex Numerical Linear Algebra problems.

Special Matrix Factorizations
---------------------------------

rocsolver_<type>potf2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotf2
.. doxygenfunction:: rocsolver_spotf2

rocsolver_<type>potf2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotf2_batched
.. doxygenfunction:: rocsolver_spotf2_batched

rocsolver_<type>potf2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotf2_strided_batched
.. doxygenfunction:: rocsolver_spotf2_strided_batched

rocsolver_<type>potrf()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotrf
.. doxygenfunction:: rocsolver_spotrf

rocsolver_<type>potrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotrf_batched
.. doxygenfunction:: rocsolver_spotrf_batched

rocsolver_<type>potrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotrf_strided_batched
.. doxygenfunction:: rocsolver_spotrf_strided_batched


General Matrix Factorizations
------------------------------

rocsolver_<type>getf2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2
.. doxygenfunction:: rocsolver_cgetf2
.. doxygenfunction:: rocsolver_dgetf2
.. doxygenfunction:: rocsolver_sgetf2

rocsolver_<type>getf2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_batched
.. doxygenfunction:: rocsolver_cgetf2_batched
.. doxygenfunction:: rocsolver_dgetf2_batched
.. doxygenfunction:: rocsolver_sgetf2_batched

rocsolver_<type>getf2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetf2_strided_batched
.. doxygenfunction:: rocsolver_cgetf2_strided_batched
.. doxygenfunction:: rocsolver_dgetf2_strided_batched
.. doxygenfunction:: rocsolver_sgetf2_strided_batched

rocsolver_<type>getrf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf
.. doxygenfunction:: rocsolver_cgetrf
.. doxygenfunction:: rocsolver_dgetrf
.. doxygenfunction:: rocsolver_sgetrf

rocsolver_<type>getrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_batched
.. doxygenfunction:: rocsolver_cgetrf_batched
.. doxygenfunction:: rocsolver_dgetrf_batched
.. doxygenfunction:: rocsolver_sgetrf_batched

rocsolver_<type>getrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrf_strided_batched
.. doxygenfunction:: rocsolver_cgetrf_strided_batched
.. doxygenfunction:: rocsolver_dgetrf_strided_batched
.. doxygenfunction:: rocsolver_sgetrf_strided_batched

rocsolver_<type>geqr2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqr2
.. doxygenfunction:: rocsolver_sgeqr2

rocsolver_<type>geqr2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqr2_batched
.. doxygenfunction:: rocsolver_sgeqr2_batched

rocsolver_<type>geqr2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqr2_strided_batched
.. doxygenfunction:: rocsolver_sgeqr2_strided_batched

rocsolver_<type>geqrf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqrf
.. doxygenfunction:: rocsolver_sgeqrf

rocsolver_<type>geqrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqrf_batched
.. doxygenfunction:: rocsolver_sgeqrf_batched

rocsolver_<type>geqrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgeqrf_strided_batched
.. doxygenfunction:: rocsolver_sgeqrf_strided_batched

rocsolver_<type>gelq2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelq2
.. doxygenfunction:: rocsolver_sgelq2

rocsolver_<type>gelq2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelq2_batched
.. doxygenfunction:: rocsolver_sgelq2_batched

rocsolver_<type>gelq2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelq2_strided_batched
.. doxygenfunction:: rocsolver_sgelq2_strided_batched

rocsolver_<type>gelqf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelqf
.. doxygenfunction:: rocsolver_sgelqf

rocsolver_<type>gelqf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelqf_batched
.. doxygenfunction:: rocsolver_sgelqf_batched

rocsolver_<type>gelqf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgelqf_strided_batched
.. doxygenfunction:: rocsolver_sgelqf_strided_batched

General systems solvers
--------------------------

rocsolver_<type>getrs()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs
.. doxygenfunction:: rocsolver_cgetrs
.. doxygenfunction:: rocsolver_dgetrs
.. doxygenfunction:: rocsolver_sgetrs

rocsolver_<type>getrs_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs_batched
.. doxygenfunction:: rocsolver_cgetrs_batched
.. doxygenfunction:: rocsolver_dgetrs_batched
.. doxygenfunction:: rocsolver_sgetrs_batched

rocsolver_<type>getrs_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_zgetrs_strided_batched
.. doxygenfunction:: rocsolver_cgetrs_strided_batched
.. doxygenfunction:: rocsolver_dgetrs_strided_batched
.. doxygenfunction:: rocsolver_sgetrs_strided_batched



Auxiliaries
=========================

rocSOLVER auxiliary functions are aliases of rocBLAS auxiliary functions. See rocBLAS auxiliary functions 
`here <https://rocblas.readthedocs.io/en/latest/api.html#auxiliary>`_.

rocSOLVER handle auxiliaries
------------------------------

rocsolver_create_handle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_create_handle

rocsolver_destroy_handle()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_destroy_handle

rocsolver_add_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_add_stream

rocsolver_set_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_stream

rocsolver_get_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_stream

Other auxiliaries
------------------------

rocsolver_set_vector()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_vector

rocsolver_get_vector()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_vector

rocsolver_set_matrix()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_set_matrix

rocsolver_get_matrix()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_get_matrix
