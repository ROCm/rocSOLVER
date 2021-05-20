.. _memory_label:

*******************************
Memory Model
*******************************

Almost all LAPACK and rocSOLVER routines require workspace memory in order to compute their results.
In contrast to LAPACK, however, pointers to the workspace are not explicitly passed to rocSOLVER
functions as arguments; instead, they are managed behind-the-scenes using a configurable device memory
model.

rocSOLVER makes use of and is integrated with `rocBLAS's memory model <https://rocblas.readthedocs.io/en/latest/device_memory.html>`_.
Workspace memory, and the scheme used to manage it, is tracked on a per-``rocblas_handle`` basis, and
the same functionality that is used to manipulate rocBLAS's workspace memory can and will also affect
rocSOLVER's workspace memory.

There are 4 schemes for device memory management:

* Automatic (managed by rocSOLVER/rocBLAS): The default scheme. Device memory persists between function
  calls and will be automatically reallocated if more memory is required by a function.
* User-managed (preallocated): The desired workspace size is specified by the user as an environment
  variable before handle creation, and cannot be altered after the handle is created.
* User-managed (manual): The desired workspace size can be manipulated using rocBLAS helper functions.
* User-owned: The user manually allocates device memory and calls a rocBLAS helper function to use it
  as the workspace.

.. toctree::
   :maxdepth: 1

   userguide_memory_automatic
   userguide_memory_usermanaged
   userguide_memory_userowned



