
*************
Introduction
*************

.. toctree::
   :maxdepth: 4 
   :caption: Contents:

Overview
==================

An implementation of `Lapack routines <http://www.netlib.org/lapack/explore-html/index.html>`_ 
on top of AMD’s Radeon Open Compute Platform runtime and toolchains (`ROCm <https://rocm.github.io>`_). 
rocSOLVER is implemented in the `HIP programming language <https://github.com/ROCm-Developer-Tools/HIP>`_
an optimized for AMD’s latest discrete GPUs.  

Brief description and functionality
====================================

rocSolver Library is in early stages of active development. New features and functionality is being continuosly added. New 
functionality is documented at each release of the ROCm platform. 

The following table summarizes the LAPACK functionality implemented in rocSOLVER's latest release.

=============================== ====== ====== ============== ==============
Lapack Auxiliary Function       single double single complex double complex
=============================== ====== ====== ============== ==============
**rocsolver_lacgv**                              x              x
**rocsolver_laswp**             x      x         x              x
**rocsolver_larfg**             x      x         x              x
**rocsolver_larft**             x      x         x              x
**rocsolver_larf**              x      x         x              x
**rocsolver_larfb**             x      x         x              x
**rocsolver_labrd**             x      x         x              x
**rocsolver_org2r**             x      x      
**rocsolver_orgqr**             x      x      
**rocsolver_orgl2**             x      x      
**rocsolver_orglq**             x      x      
**rocsolver_orgbr**             x      x      
**rocsolver_orm2r**             x      x      
**rocsolver_ormqr**             x      x      
**rocsolver_orml2**             x      x      
**rocsolver_ormlq**             x      x      
**rocsolver_ormbr**             x      x      
**rocsolver_ung2r**                              x              x      
**rocsolver_ungqr**                              x              x     
**rocsolver_ungl2**                              x              x      
**rocsolver_unglq**                              x              x     
**rocsolver_ungbr**                              x              x     
**rocsolver_unm2r**                              x              x      
**rocsolver_unmqr**                              x              x     
**rocsolver_unml2**                              x              x      
**rocsolver_unmlq**                              x              x     
**rocsolver_unmbr**                              x              x     
=============================== ====== ====== ============== ==============

=============================== ====== ====== ============== ==============
Lapack Function                 single double single complex double complex
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
**rocsolver_gelq2**             x      x          x             x
rocsolver_gelq2_batched         x      x          x             x
rocsolver_gelq2_strided_batched x      x          x             x
**rocsolver_gelqf**             x      x          x             x
rocsolver_gelqf_batched         x      x          x             x
rocsolver_gelqf_strided_batched x      x          x             x
**rocsolver_getrs**             x      x          x             x
rocsolver_getrs_batched         x      x          x             x
rocsolver_getrs_strided_batched x      x          x             x
**rocsolver_getri**             x      x          x             x
rocsolver_getri_batched         x      x          x             x
rocsolver_getri_strided_batched x      x          x             x
**rocsolver_gebd2**             x      x          x             x 
rocsolver_gebd2_batched         x      x          x             x
rocsolver_gebd2_strided_batched x      x          x             x
**rocsolver_gebrd**             x      x          x             x
rocsolver_gebrd_batched         x      x          x             x
rocsolver_gebrd_strided_batched x      x          x             x
=============================== ====== ====== ============== ==============


 
