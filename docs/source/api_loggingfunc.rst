***************************
Logging and Other Functions
***************************

.. _api_logging:

Logging functions
===============================

These are functions that enable and control rocSOLVER's :ref:`logging-label` capabilities. Functions
are divided in two categories:

* :ref:`initialize` functions. Used to initialize and terminate the logging.
* :ref:`profile` functions. Provide functionality for the profile logging mode.


.. _initialize:

Logging set-up and tear-down
-------------------------------

.. contents:: List of logging initialization functions
   :local:
   :backlinks: top

rocsolver_log_begin()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_begin

rocsolver_log_end()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_end

rocsolver_log_set_layer_mode()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_set_layer_mode

rocsolver_log_set_max_levels()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_set_max_levels

rocsolver_log_restore_defaults()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_restore_defaults



.. _profile:

Profile logging
------------------------------

.. contents:: List of profile logging functions
   :local:
   :backlinks: top

rocsolver_log_write_profile()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_write_profile

rocsolver_log_flush_profile()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenfunction:: rocsolver_log_flush_profile



.. _libraryinfo:

Library information
===============================

.. contents:: List of library information functions
   :local:
   :backlinks: top

rocsolver_get_version_string()
------------------------------------
.. doxygenfunction:: rocsolver_get_version_string

rocsolver_get_version_string_size()
------------------------------------
.. doxygenfunction:: rocsolver_get_version_string_size

