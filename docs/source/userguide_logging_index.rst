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
   :maxdepth: 1

   userguide_logging_modes
   userguide_logging_initialize
   userguide_logging_multiplethreads



