.. _logging-label:

*************************
Multi-level Logging
*************************

Similar to `rocBLAS logging <https://rocblas.readthedocs.io/en/latest/logging.html>`_,
rocSOLVER provides logging facilities that can be used to output information
on rocSOLVER function calls. Three modes of logging are supported: trace logging, bench logging,
and profile logging.

Note that performance will degrade when logging is enabled.

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top


Logging modes
================================================

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


Initialization and set-up
================================================

In order to use rocSOLVER's logging facilities, the user must first call ``rocsolver_log_begin``
in order to allocate the internal data structures used for logging and begin the logging session.
The user may then specify a layer mode and max level depth, either programmatically using
``rocsolver_log_set_layer_mode``, ``rocsolver_log_set_max_levels``, or by setting the corresponding
environment variables.

The layer mode specifies which logging type(s) are activated, and can be ``rocblas_layer_mode_none``,
``rocblas_layer_mode_log_trace``, ``rocblas_layer_mode_log_bench``, ``rocblas_layer_mode_log_profile``,
or a bitwise combination of these. The max level depth specifies the default maximum depth of nested
function calls that may appear in the trace and profile logging.

Both the default layer mode and max level depth can be specified using environment variables.

* ``ROCSOLVER_LAYER``
* ``ROCSOLVER_LEVELS``

If these variables are not set, the layer mode will default to ``rocblas_layer_mode_none`` and the
max level depth will default to 1. These defaults can be restored by calling the function
``rocsolver_log_restore_defaults``.

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

The results of profile logging, if enabled, can be printed using ``rocsolver_log_write_profile``
or ``rocsolver_log_flush_profile``. Once logging facilities are no longer required (e.g. at
program termination), the user must call ``rocsolver_log_end`` to free the data structures used
for logging. If the profile log has not been flushed beforehand, then ``rocsolver_log_end``
will also output the results of profile logging.

For more details on the mentioned logging functions, see the :ref:`Logging functions section <api_logging>`
on the rocSOLVER API document.


Multiple host threads
================================================

The logging facilities for rocSOLVER assume that each ``rocblas_handle`` is associated with at
most one host thread. When using rocSOLVER's multi-level logging setup, it is recommended to
create a separate ``rocblas_handle`` for each host thread.

The rocsolver_log_* functions are not thread-safe. Calling a log function while any rocSOLVER
routine is executing on another host thread will result in undefined behaviour. Once enabled,
logging data collection is thread-safe. However, note that trace logging will likely result in
garbled trace trees if rocSOLVER routines are called from multiple host threads.

