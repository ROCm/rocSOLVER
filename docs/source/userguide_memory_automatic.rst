
Automatic workspace 
================================================

.. toctree::

By default, rocSOLVER will automatically allocate device memory to be used as internal workspace
using the rocBLAS memory model, and will increase the amount of allocated memory as needed by rocSOLVER
functions. If this scheme is in use, the function ``rocblas_is_managing_device_memory`` will return
``true``. In order to re-enable this scheme if it is not in use, a ``nullptr`` or zero size can be
passed to the helper functions ``rocblas_set_device_memory_size`` or ``rocblas_set_workspace`` (see
examples below).

This scheme has the disadvantage that automatic reallocation is synchronizing, and the user cannot
control when this synchronization happens.
