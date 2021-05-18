
*************
Introduction
*************

.. toctree::
   :maxdepth: 4
   :caption: Contents:

Overview
==================

rocSOLVER is an implementation of `LAPACK routines <https://www.netlib.org/lapack/explore-html/modules.html>`_
on top of `AMD ROCm <https://rocm.github.io>`_. rocSOLVER is implemented in the
`HIP programming language <https://github.com/ROCm-Developer-Tools/HIP>`_ and optimized for AMD's
latest discrete GPUs.

Brief description and functionality
====================================

The rocSOLVER library is in the early stages of active development. New features are being
continuously added, with new functionality documented at each release of the ROCm platform.

The following table summarizes the LAPACK functionality implemented in rocSOLVER's latest release.

=============================== ====== ====== ============== ==============
LAPACK Auxiliary Function       single double single complex double complex
=============================== ====== ====== ============== ==============
**rocsolver_lacgv**                              x              x
**rocsolver_laswp**             x      x         x              x
**rocsolver_larfg**             x      x         x              x
**rocsolver_larft**             x      x         x              x
**rocsolver_larf**              x      x         x              x
**rocsolver_larfb**             x      x         x              x
**rocsolver_labrd**             x      x         x              x
**rocsolver_latrd**             x      x         x              x
**rocsolver_bdsqr**             x      x         x              x
**rocsolver_org2r**             x      x
**rocsolver_orgqr**             x      x
**rocsolver_orgl2**             x      x
**rocsolver_orglq**             x      x
**rocsolver_org2l**             x      x
**rocsolver_orgql**             x      x
**rocsolver_orgbr**             x      x
**rocsolver_orgtr**             x      x
**rocsolver_orm2r**             x      x
**rocsolver_ormqr**             x      x
**rocsolver_orml2**             x      x
**rocsolver_ormlq**             x      x
**rocsolver_orm2l**             x      x
**rocsolver_ormql**             x      x
**rocsolver_ormbr**             x      x
**rocsolver_ormtr**             x      x
**rocsolver_ung2r**                              x              x
**rocsolver_ungqr**                              x              x
**rocsolver_ungl2**                              x              x
**rocsolver_unglq**                              x              x
**rocsolver_ung2l**                              x              x
**rocsolver_ungql**                              x              x
**rocsolver_ungbr**                              x              x
**rocsolver_ungtr**                              x              x
**rocsolver_unm2r**                              x              x
**rocsolver_unmqr**                              x              x
**rocsolver_unml2**                              x              x
**rocsolver_unmlq**                              x              x
**rocsolver_unm2l**                              x              x
**rocsolver_unmql**                              x              x
**rocsolver_unmbr**                              x              x
**rocsolver_unmtr**                              x              x
**rocsolver_sterf**             x      x
**rocsolver_steqr**             x      x         x              x
**rocsolver_stedc**             x      x         x              x
=============================== ====== ====== ============== ==============

=============================== ====== ====== ============== ==============
LAPACK Function                 single double single complex double complex
=============================== ====== ====== ============== ==============
**rocsolver_potf2**             x      x          x             x
rocsolver_potf2_batched         x      x          x             x
rocsolver_potf2_strided_batched x      x          x             x
**rocsolver_potrf**             x      x          x             x
rocsolver_potrf_batched         x      x          x             x
rocsolver_potrf_strided_batched x      x          x             x
**rocsolver_getf2**             x      x          x             x
rocsolver_getf2_batched         x      x          x             x
rocsolver_getf2_strided_batched x      x          x             x
**rocsolver_getrf**             x      x          x             x
rocsolver_getrf_batched         x      x          x             x
rocsolver_getrf_strided_batched x      x          x             x
**rocsolver_geqr2**             x      x          x             x
rocsolver_geqr2_batched         x      x          x             x
rocsolver_geqr2_strided_batched x      x          x             x
**rocsolver_geqrf**             x      x          x             x
rocsolver_geqrf_batched         x      x          x             x
rocsolver_geqrf_strided_batched x      x          x             x
**rocsolver_geql2**             x      x          x             x
rocsolver_geql2_batched         x      x          x             x
rocsolver_geql2_strided_batched x      x          x             x
**rocsolver_geqlf**             x      x          x             x
rocsolver_geqlf_batched         x      x          x             x
rocsolver_geqlf_strided_batched x      x          x             x
**rocsolver_gelq2**             x      x          x             x
rocsolver_gelq2_batched         x      x          x             x
rocsolver_gelq2_strided_batched x      x          x             x
**rocsolver_gelqf**             x      x          x             x
rocsolver_gelqf_batched         x      x          x             x
rocsolver_gelqf_strided_batched x      x          x             x
**rocsolver_getrs**             x      x          x             x
rocsolver_getrs_batched         x      x          x             x
rocsolver_getrs_strided_batched x      x          x             x
**rocsolver_trtri**             x      x          x             x
rocsolver_trtri_batched         x      x          x             x
rocsolver_trtri_strided_batched x      x          x             x
**rocsolver_getri**             x      x          x             x
rocsolver_getri_batched         x      x          x             x
rocsolver_getri_strided_batched x      x          x             x
**rocsolver_gels**              x      x          x             x
rocsolver_gels_batched          x      x          x             x
rocsolver_gels_strided_batched  x      x          x             x
**rocsolver_gebd2**             x      x          x             x
rocsolver_gebd2_batched         x      x          x             x
rocsolver_gebd2_strided_batched x      x          x             x
**rocsolver_gebrd**             x      x          x             x
rocsolver_gebrd_batched         x      x          x             x
rocsolver_gebrd_strided_batched x      x          x             x
**rocsolver_gesvd**             x      x          x             x
rocsolver_gesvd_batched         x      x          x             x
rocsolver_gesvd_strided_batched x      x          x             x
**rocsolver_sytd2**             x      x
rocsolver_sytd2_batched         x      x
rocsolver_sytd2_strided_batched x      x
**rocsolver_sytrd**             x      x
rocsolver_sytrd_batched         x      x
rocsolver_sytrd_strided_batched x      x
**rocsolver_hetd2**                               x             x
rocsolver_hetd2_batched                           x             x
rocsolver_hetd2_strided_batched                   x             x
**rocsolver_hetrd**                               x             x
rocsolver_hetrd_batched                           x             x
rocsolver_hetrd_strided_batched                   x             x
**rocsolver_sygs2**             x      x
rocsolver_sygs2_batched         x      x
rocsolver_sygs2_strided_batched x      x
**rocsolver_sygst**             x      x
rocsolver_sygst_batched         x      x
rocsolver_sygst_strided_batched x      x
**rocsolver_hegs2**                               x             x
rocsolver_hegs2_batched                           x             x
rocsolver_hegs2_strided_batched                   x             x
**rocsolver_hegst**                               x             x
rocsolver_hegst_batched                           x             x
rocsolver_hegst_strided_batched                   x             x
**rocsolver_syev**              x      x
rocsolver_syev_batched          x      x
rocsolver_syev_strided_batched  x      x
**rocsolver_heev**                                x             x
rocsolver_heev_batched                            x             x
rocsolver_heev_strided_batched                    x             x
**rocsolver_sygv**              x      x
rocsolver_sygv_batched          x      x
rocsolver_sygv_strided_batched  x      x
**rocsolver_hegv**                                x             x
rocsolver_hegv_batched                            x             x
rocsolver_hegv_strided_batched                    x             x
=============================== ====== ====== ============== ==============

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


