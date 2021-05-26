
User-managed workspace 
================================================

.. toctree::

Alternatively, the user can manually specify an amount of memory to be allocated by rocSOLVER/rocBLAS.
This allows the user to control when and if memory is reallocated and synchronization occurs. However,
function calls will fail if there is not enough allocated memory.

Minimum required size
------------------------------

In order to choose an appropriate amount of memory to allocate, rocSOLVER can be queried to determine
the minimum amount of memory required for functions to complete. The query can be started by calling
``rocblas_start_device_memory_size_query``, followed by calls to the desired functions with appropriate
problem sizes (a null pointer can be passed to the device pointer arguments). A final call to
``rocblas_stop_device_memory_size_query`` will return the minimum required size.

For example, the following code snippet will return the memory size required to solve a 1024*1024 linear
system with 1 right-hand side (involving calls to ``getrf`` and ``getrs``):

.. code-block:: cpp

    size_t memory_size;
    rocblas_start_device_memory_size_query(handle);
    rocsolver_dgetrf(handle, 1024, 1024, nullptr, lda, nullptr, nullptr);
    rocsolver_dgetrs(handle, rocblas_operation_none, 1024, 1, nullptr, lda, nullptr, nullptr, ldb);
    rocblas_stop_device_memory_size_query(handle, &memory_size);

For more details on the rocBLAS APIs, see the 
`rocBLAS documentation <https://rocblas.readthedocs.io/en/latest/functions.html#device-memory-functions>`_.


Using an environment variable
------------------------------

The desired workspace size can be provided before creation of the ``rocblas_handle`` by setting the
value of environment variable ``ROCBLAS_DEVICE_MEMORY_SIZE``. If this variable is unset or the value
is == 0, then it will be ignored. Note that a workspace size set in this way cannot be changed once
the handle has been created.

Using helper functions
------------------------------

Another way to set the desired workspace size is by using the helper function ``rocblas_set_device_memory_size``.
This function is called after handle creation and can be called multiple times; however, it is
recommended to first synchronize the handle stream if a rocSOLVER or rocBLAS routine has already been
called. For example:

.. code-block:: cpp

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    hipStreamSynchronize(stream);

    rocblas_set_device_memory_size(handle, memory_size);

For more details on the rocBLAS APIs, see the 
`rocBLAS documentation <https://rocblas.readthedocs.io/en/latest/functions.html#device-memory-functions>`_.
