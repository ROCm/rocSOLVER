
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

*************
Introduction
*************

An implementation of Lapack routines on top of AMD’s Radeon Open Compute Platform (ROCm) runtime and toolchains. 
rocSOLVER is implemented in the HIP programming language; it is based on rocBLAS, an optimized BLAS 
implementation for AMD’s latest discrete GPUs. More information about rocBLAS can be found 
`here <https://rocblas.readthedocs.io/en/latest/index.html>`_.

Build and install
===================

rocSOLVER requires `cmake <https://cmake.org/install/>`_ 
and `ROCm <https://rocm.github.io/install.html>`_, including 
`hip <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`_ and 
`rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_, to be installed. 

Once these requirements are satisfied, the following
instructions will build and install rocSOLVER:

.. code-block:: bash
   
    mkdir build && cd build
    CXX=/opt/rocm/bin/hcc cmake ..
    make
    make install

  
