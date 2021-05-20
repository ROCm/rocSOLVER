
Prerequisites
=================

.. toctree::

rocSOLVER requires a ROCm-enabled platform. For more information, see the
`ROCm install guide <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_.

rocSOLVER also requires a compatible version of rocBLAS installed on the system.
For more information, see the `rocBLAS install guide <https://rocblas.readthedocs.io/en/master/install.html>`_.

rocBLAS and rocSOLVER are both still under active development, and it is hard to define minimal
compatibility versions. For now, a good rule of thumb is to always use rocSOLVER together with the
matching rocBLAS version. For example, if you want to install rocSOLVER from ROCm 3.3 release, then
be sure that ROCm 3.3 rocBLAS is also installed; if you are building the rocSOLVER branch tip, then
you will need to build and install rocBLAS branch tip as well.

