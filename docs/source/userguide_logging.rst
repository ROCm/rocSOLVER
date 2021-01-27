
*************************
Multi-level Logging
*************************

.. toctree::
   :maxdepth: 4
   :caption: Contents:

Similar to rocBLAS, rocSOLVER provides logging facilities that can be used to output information
on rocSOLVER function calls. Three types of logging are supported: trace logging, bench logging,
and profile logging.

Note that performance will degrade when logging is enabled.

Logging types
================================================

Trace logging outputs a line each time an internal rocSOLVER or rocBLAS routine is called,
outputting the function name and the values of its arguments (excluding stride arguments). The
maximum depth of nested function calls that can appear in the log is specified by the user.

Bench logging outputs a line each time a public rocSOLVER routine is called (excluding
auxiliary library functions), outputting a line that can be used with the executable
``rocsolver-bench`` to call the function with the same size arguments.

Profile logging, upon terminating the logging facility using ``rocsolver_logging_cleanup``,
will output statistics on each called internal rocSOLVER and rocBLAS routine. These include the
number of times each function was called, the total program runtime occupied by the function,
and the total program runtime occupied by its nested function calls. Note that, when profile
logging is enabled, the stream will be synchronized after every internal function call.


Initialization and set-up
================================================

In order to use rocSOLVER's logging facilities, the user must first call ``rocsolver_logging_initialize``
with a default layer mode and max level depth. The layer mode specifies which logging type(s) are
activated by default, and can be ``rocblas_layer_mode_none``, ``rocblas_layer_mode_log_trace``,
``rocblas_layer_mode_log_bench``, ``rocblas_layer_mode_log_profile``, or a bitwise combination
of these. The max level depth specifies the default maximum depth of nested function calls that
may appear in the trace logging.

Both the default layer mode and max level depth can be overridden using two environment variables:

* ``ROCSOLVER_LAYER``
* ``ROCSOLVER_LEVELS``

``ROCSOLVER_LAYER`` is a bitwise OR of zero or more bit masks as follows:

*  If ``ROCSOLVER_LAYER`` is not set, then there is no logging
*  If ``(ROCSOLVER_LAYER & 1) != 0``, then there is trace logging
*  If ``(ROCSOLVER_LAYER & 2) != 0``, then there is bench logging
*  If ``(ROCSOLVER_LAYER & 4) != 0``, then there is profile logging

Three environment variables can set the full path name for a log file:

* ``ROCSOLVER_LOG_TRACE_PATH`` sets the full path name for trace logging
* ``ROCSOLVER_LOG_BENCH_PATH`` sets the full path name for bench logging
* ``ROCSOLVER_LOG_PROFILE_PATH`` sets the full path name for profile logging

If one of these environment variables is not set, then ``ROCSOLVER_LOG_PATH`` sets the full path
for the corresponding logging, if it is set. If neither the above nor ``ROCSOLVER_LOG_PATH`` are
set, then the corresponding logging output is streamed to standard error.

Once logging facilities are no longer required (e.g. at program termination), the user must
call ``rocsolver_logging_cleanup``.

Note that when profile logging is enabled, memory usage will increase. If the program exits
without calling ``rocsolver_logging_cleanup``, then profile logging will not be outputted
before the program exits.

