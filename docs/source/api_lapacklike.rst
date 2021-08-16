
***********************
Lapack-like Functions
***********************

Other Lapack-like routines provided by rocSOLVER. These are divided into the following subcategories:

* :ref:`liketriangular`. Based on Gaussian elimination.
* :ref:`likelinears`. Based on triangular factorizations.

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



.. _liketriangular:

Triangular factorizations
===========================

.. contents:: List of Lapack-like triangular factorizations
   :local:
   :backlinks: top

.. _getf2_npvt:

rocsolver_<type>getf2_npvt()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt

rocsolver_<type>getf2_npvt_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_batched

rocsolver_<type>getf2_npvt_strided_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetf2_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetf2_npvt_strided_batched

.. _getrf_npvt:

rocsolver_<type>getrf_npvt()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt

rocsolver_<type>getrf_npvt_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_batched

rocsolver_<type>getrf_npvt_strided_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetrf_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetrf_npvt_strided_batched



.. _likelinears:

Linear-systems solvers
========================

.. contents:: List of Lapack-like linear solvers
   :local:
   :backlinks: top

.. _getri_npvt:

rocsolver_<type>getri_npvt()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt

rocsolver_<type>getri_npvt_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_batched

rocsolver_<type>getri_npvt_strided_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_strided_batched

.. _getri_outofplace:

rocsolver_<type>getri_outofplace()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace

rocsolver_<type>getri_outofplace_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace_batched

rocsolver_<type>getri_outofplace_strided_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_outofplace_strided_batched

.. _getri_npvt_outofplace:

rocsolver_<type>getri_npvt_outofplace()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace

rocsolver_<type>getri_npvt_outofplace_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace_batched

rocsolver_<type>getri_npvt_outofplace_strided_batched()
-----------------------------------------------------
.. doxygenfunction:: rocsolver_zgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_cgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_dgetri_npvt_outofplace_strided_batched
   :outline:
.. doxygenfunction:: rocsolver_sgetri_npvt_outofplace_strided_batched

