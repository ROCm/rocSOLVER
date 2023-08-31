
*****************************************
Re-factorization and direct solvers
*****************************************

These are functions that implement direct solvers for sparse systems with
different coefficient matrices that share the same sparsity pattern.
The re-factorization functions are divided into the following categories:

* :ref:`rfinit`. Basic functions to initialize and destroy meta data.
* :ref:`rfrefact`. Re-factorization of new matrices given a known sparsity pattern.
* :ref:`rfsolver`. Based on triangular re-factorization.

.. note::
    Throughout the APIs' descriptions, we use the following notations:

    * x[i] stands for the i-th element of vector x, while A[i,j] represents the element
      in the i-th row and j-th column of matrix A. Indices are 1-based, i.e. x[1] is the first
      element of x.
    * If X is a real vector or matrix, :math:`X^T` indicates its transpose; if X is complex, then
      :math:`X^H` represents its conjugate transpose. When X could be real or complex, we use X' to
      indicate X transposed or X conjugate transposed, accordingly.
    * x_i :math:`=x_i`; we sometimes use both notations, :math:`x_i` when displaying mathematical
      equations, and x_i in the text describing the function parameters.



.. _rfinit:

Initialization and meta data
==================================

.. contents:: List of initialization functions
   :local:
   :backlinks: top


.. _rfinfocreate:

rocsolver_create_rfinfo()
---------------------------------------
.. doxygenfunction:: rocsolver_create_rfinfo


.. _rfinfodestroy:

rocsolver_destroy_rfinfo()
---------------------------------------
.. doxygenfunction:: rocsolver_destroy_rfinfo


.. _rfinfoset:

rocsolver_set_rfinfo_mode()
---------------------------------------
.. doxygenfunction:: rocsolver_set_rfinfo_mode


.. _rfinfoget:

rocsolver_get_rfinfo_mode()
---------------------------------------
.. doxygenfunction:: rocsolver_get_rfinfo_mode


.. _rfanalysis:

rocsolver_csrrf_analysis()
--------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_analysis
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_analysis



.. _rfrefact:

Triangular re-factorization
==================================

.. contents:: List of re-factorization functions
   :local:
   :backlinks: top

.. _rfsumlu:

rocsolver_<type>csrrf_sumlu()
----------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_sumlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_sumlu


.. _rfsplitlu:

rocsolver_<type>csrrf_splitlu()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_splitlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_splitlu


.. _rfrefactlu:

rocsolver_<type>csrrf_refactlu()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_refactlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_refactlu


.. _rfrefactchol:

rocsolver_<type>csrrf_refactchol()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_refactchol
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_refactchol



.. _rfsolver:

Direct sparse solvers
==================================

.. contents:: List of direct solvers
   :local:
   :backlinks: top

.. _rfsolve:

rocsolver_<type>csrrf_solve()
----------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_solve
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_solve
