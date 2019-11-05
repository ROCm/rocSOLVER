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

All rocSOLVER types are aliases of rocBLAS types. 
See rocBLAS types `here <https://rocblas.readthedocs.io/en/latest/api.html#types>`_.

rocsolver_int
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_int

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

rocsolver_status
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocsolver_status



Lapack Auxiliary Functions
============================

rocsolver_<type>laswp()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlaswp
.. doxygenfunction:: rocsolver_slaswp

rocsolver_<type>larfg()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarfg
.. doxygenfunction:: rocsolver_slarfg

rocsolver_<type>larf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dlarf
.. doxygenfunction:: rocsolver_slarf



Special Matrix Factorizations
=================================

rocsolver_<type>potf2()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dpotf2
.. doxygenfunction:: rocsolver_spotf2


General Matrix Factorizations
==============================

rocsolver_<type>getf2()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetf2
.. doxygenfunction:: rocsolver_sgetf2

rocsolver_<type>getf2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetf2_batched
.. doxygenfunction:: rocsolver_sgetf2_batched

rocsolver_<type>getf2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetf2_strided_batched
.. doxygenfunction:: rocsolver_sgetf2_strided_batched

rocsolver_<type>getrf()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetrf
.. doxygenfunction:: rocsolver_sgetrf

rocsolver_<type>getrf_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetrf_batched
.. doxygenfunction:: rocsolver_sgetrf_batched

rocsolver_<type>getrf_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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



General systems solvers
===============================

rocsolver_<type>getrs()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocsolver_dgetrs
.. doxygenfunction:: rocsolver_sgetrs



Auxiliaries
=========================

rocSOLVER auxiliary functions are aliases of rocBLAS auxiliary functions. See rocBLAS auxiliary functions 
`here <https://rocblas.readthedocs.io/en/latest/api.html#auxiliary>`_.

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
