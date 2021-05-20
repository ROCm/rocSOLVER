
Multiple host threads
================================================

.. toctree::

The logging facilities for rocSOLVER assume that each ``rocblas_handle`` is associated with at
most one host thread. When using rocSOLVER's multi-level logging setup, it is recommended to
create a separate ``rocblas_handle`` for each host thread.

The rocsolver_log_* functions are not thread-safe. Calling a log function while any rocSOLVER
routine is executing on another host thread will result in undefined behaviour. Once enabled,
logging data collection is thread-safe. However, note that trace logging will likely result in
garbled trace trees if rocSOLVER routines are called from multiple host threads.
