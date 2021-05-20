
User-owned workspace 
================================================

.. toctree::

Finally, the user may opt to manage the workspace memory manually using HIP. By calling the function
``rocblas_set_workspace``, the user may pass a pointer to device memory to rocBLAS that will be used
as the workspace for rocSOLVER. For example:

.. code-block:: cpp

    void* device_memory;
    hipMalloc(&device_memory, memory_size);
    rocblas_set_workspace(handle, device_memory, memory_size);

    // perform computations here

    rocblas_set_workspace(handle, nullptr, 0);
    hipFree(device_memory);
