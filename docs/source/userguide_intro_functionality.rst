
Currently implemented functionality
====================================

.. toctree::

The rocSOLVER library is in the early stages of active development. New features are being
continuously added, with new functionality documented at each `release of the ROCm platform <https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html>`_.

The following tables summarizes the LAPACK functionality implemented for the different supported precisions in rocSOLVER's latest release.
All the LAPACK main functions include *_batched* and *_strided_batched* versions. For a complete description of the listed 
routines, please see the :ref:`rocSOLVER API <library_api>` document.

LAPACK auxiliary functions
----------------------------

.. csv-table:: Vector and matrix manipulations
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_lacgv <lacgv>`, x, x, x, x
    :ref:`rocsolver_laswp <laswp>`, x, x, x, x

.. csv-table:: Householder reflections
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_larfg <larfg>`, x, x, x, x
    :ref:`rocsolver_larf <larf>`, x, x, x, x
    :ref:`rocsolver_larft <larft>`, x, x, x, x
    :ref:`rocsolver_larfb <larfb>`, x, x, x, x

.. csv-table:: Bidiagonal forms
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_labrd <labrd>`, x, x, x, x
    :ref:`rocsolver_bdsqr <bdsqr>`, x, x, x, x

.. csv-table:: Tridiagonal forms
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_latrd <latrd>`, x, x, x, x
    :ref:`rocsolver_sterf <sterf>`, x, x, , 
    :ref:`rocsolver_steqr <steqr>`, x, x, x, x
    :ref:`rocsolver_stedc <stedc>`, x, x, x, x

.. csv-table:: Orthonormal matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_org2r <org2r>`, x, x, , 
    :ref:`rocsolver_orgqr <orgqr>`, x, x, ,
    :ref:`rocsolver_orgl2 <orgl2>`, x, x, , 
    :ref:`rocsolver_orglq <orglq>`, x, x, , 
    :ref:`rocsolver_org2l <org2l>`, x, x, , 
    :ref:`rocsolver_orgql <orgql>`, x, x, , 
    :ref:`rocsolver_orgbr <orgbr>`, x, x, , 
    :ref:`rocsolver_orgtr <orgtr>`, x, x, , 
    :ref:`rocsolver_orm2r <orm2r>`, x, x, , 
    :ref:`rocsolver_ormqr <ormqr>`, x, x, , 
    :ref:`rocsolver_orml2 <orml2>`, x, x, , 
    :ref:`rocsolver_ormlq <ormlq>`, x, x, , 
    :ref:`rocsolver_orm2l <orm2l>`, x, x, , 
    :ref:`rocsolver_ormql <ormql>`, x, x, , 
    :ref:`rocsolver_ormbr <ormbr>`, x, x, ,
    :ref:`rocsolver_ormtr <ormtr>`, x, x, , 

.. csv-table:: Unitary matrices
    :header: "Function", "single", "double", "single complex", "double complex"

    :ref:`rocsolver_ung2r <ung2r>`, , , x, x
    :ref:`rocsolver_ungqr <ungqr>`, , , x, x
    :ref:`rocsolver_ungl2 <ungl2>`, , , x, x
    :ref:`rocsolver_unglq <unglq>`, , , x, x
    :ref:`rocsolver_ung2l <ung2l>`, , , x, x
    :ref:`rocsolver_ungql <ungql>`, , , x, x
    :ref:`rocsolver_ungbr <ungbr>`, , , x, x
    :ref:`rocsolver_ungtr <ungtr>`, , , x, x
    :ref:`rocsolver_unm2r <unm2r>`, , , x, x
    :ref:`rocsolver_unmqr <unmqr>`, , , x, x
    :ref:`rocsolver_unml2 <unml2>`, , , x, x
    :ref:`rocsolver_unmlq <unmlq>`, , , x, x
    :ref:`rocsolver_unm2l <unm2l>`, , , x, x
    :ref:`rocsolver_unmql <unmql>`, , , x, x
    :ref:`rocsolver_unmbr <unmbr>`, , , x, x
    :ref:`rocsolver_unmtr <unmtr>`, , , x, x





LAPACK main functions
----------------------------


================================ ====== ====== ============== ==============
LAPACK Function                  single double single complex double complex
================================ ====== ====== ============== ==============
**rocsolver_potf2**                x      x          x             x
rocsolver_potf2_batched            x      x          x             x
rocsolver_potf2_strided_batched    x      x          x             x
**rocsolver_potrf**                x      x          x             x
rocsolver_potrf_batched            x      x          x             x
rocsolver_potrf_strided_batched    x      x          x             x
**rocsolver_getf2**                x      x          x             x
rocsolver_getf2_batched            x      x          x             x
rocsolver_getf2_strided_batched    x      x          x             x
**rocsolver_getrf**                x      x          x             x
rocsolver_getrf_batched            x      x          x             x
rocsolver_getrf_strided_batched    x      x          x             x
**rocsolver_geqr2**                x      x          x             x
rocsolver_geqr2_batched            x      x          x             x
rocsolver_geqr2_strided_batched    x      x          x             x
:ref:`rocsolver_geqrf <geqrf>`     x      x          x             x

 _batched                           

 _strided_batched                   
**rocsolver_geql2**                x      x          x             x
rocsolver_geql2_batched            x      x          x             x
rocsolver_geql2_strided_batched    x      x          x             x
**rocsolver_geqlf**                x      x          x             x
rocsolver_geqlf_batched            x      x          x             x
rocsolver_geqlf_strided_batched    x      x          x             x
**rocsolver_gelq2**                x      x          x             x
rocsolver_gelq2_batched            x      x          x             x
rocsolver_gelq2_strided_batched    x      x          x             x
**rocsolver_gelqf**                x      x          x             x
rocsolver_gelqf_batched            x      x          x             x
rocsolver_gelqf_strided_batched    x      x          x             x
**rocsolver_getrs**                x      x          x             x
rocsolver_getrs_batched            x      x          x             x
rocsolver_getrs_strided_batched    x      x          x             x
**rocsolver_trtri**                x      x          x             x
rocsolver_trtri_batched            x      x          x             x
rocsolver_trtri_strided_batched    x      x          x             x
**rocsolver_getri**                x      x          x             x
rocsolver_getri_batched            x      x          x             x
rocsolver_getri_strided_batched    x      x          x             x
**rocsolver_gels**                 x      x          x             x
rocsolver_gels_batched             x      x          x             x
rocsolver_gels_strided_batched     x      x          x             x
**rocsolver_gebd2**                x      x          x             x
rocsolver_gebd2_batched            x      x          x             x
rocsolver_gebd2_strided_batched    x      x          x             x
**rocsolver_gebrd**                x      x          x             x
rocsolver_gebrd_batched            x      x          x             x
rocsolver_gebrd_strided_batched    x      x          x             x
**rocsolver_gesvd**                x      x          x             x
rocsolver_gesvd_batched            x      x          x             x
rocsolver_gesvd_strided_batched    x      x          x             x
**rocsolver_sytd2**                x      x
rocsolver_sytd2_batched            x      x
rocsolver_sytd2_strided_batched    x      x
**rocsolver_sytrd**                x      x
rocsolver_sytrd_batched            x      x
rocsolver_sytrd_strided_batched    x      x
**rocsolver_hetd2**                                  x             x
rocsolver_hetd2_batched                              x             x
rocsolver_hetd2_strided_batched                      x             x
**rocsolver_hetrd**                                  x             x
rocsolver_hetrd_batched                              x             x
rocsolver_hetrd_strided_batched                      x             x
**rocsolver_sygs2**                x      x
rocsolver_sygs2_batched            x      x
rocsolver_sygs2_strided_batched    x      x
**rocsolver_sygst**                x      x
rocsolver_sygst_batched            x      x
rocsolver_sygst_strided_batched    x      x
**rocsolver_hegs2**                                  x             x
rocsolver_hegs2_batched                              x             x
rocsolver_hegs2_strided_batched                      x             x
**rocsolver_hegst**                                  x             x
rocsolver_hegst_batched                              x             x
rocsolver_hegst_strided_batched                      x             x
**rocsolver_syev**                 x      x
rocsolver_syev_batched             x      x
rocsolver_syev_strided_batched     x      x
**rocsolver_heev**                                   x             x
rocsolver_heev_batched                               x             x
rocsolver_heev_strided_batched                       x             x
**rocsolver_sygv**                 x      x
rocsolver_sygv_batched             x      x
rocsolver_sygv_strided_batched     x      x
**rocsolver_hegv**                                   x             x
rocsolver_hegv_batched                               x             x
rocsolver_hegv_strided_batched                       x             x
================================ ====== ====== ============== ==============

=========================================== ====== ====== ============== ==============
Lapack-like Function                        single double single complex double complex
=========================================== ====== ====== ============== ==============
**rocsolver_getf2_npvt**                        x      x          x             x
rocsolver_getf2_npvt_batched                    x      x          x             x
rocsolver_getf2_npvt_strided_batched            x      x          x             x
**rocsolver_getrf_npvt**                        x      x          x             x
rocsolver_getrf_npvt_batched                    x      x          x             x
rocsolver_getrf_npvt_strided_batched            x      x          x             x
**rocsolver_getri_outofplace**                  x      x          x             x
rocsolver_getri_outofplace_batched              x      x          x             x
rocsolver_getri_outofplace_strided_batched      x      x          x             x
=========================================== ====== ====== ============== ==============

