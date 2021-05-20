
Logging Modes
================================================

.. toctree::

Trace logging
--------------

Trace logging outputs a line each time an internal rocSOLVER or rocBLAS routine is called,
outputting the function name and the values of its arguments (excluding stride arguments). The
maximum depth of nested function calls that can appear in the log is specified by the user.

Bench logging
----------------

Bench logging outputs a line each time a public rocSOLVER routine is called (excluding
auxiliary library functions), outputting a line that can be used with the executable
``rocsolver-bench`` to call the function with the same size arguments.

Profile logging
-------------------

Profile logging, upon calling ``rocsolver_log_write_profile`` or ``rocsolver_log_flush_profile``,
or terminating the logging session using ``rocsolver_log_end``, will output statistics on each
called internal rocSOLVER and rocBLAS routine. These include the number of times each function
was called, the total program runtime occupied by the function, and the total program runtime
occupied by its nested function calls. As with trace logging, the maximum depth of nested output
is specified by the user. Note that, when profile logging is enabled, the stream will be synchronized
after every internal function call.
