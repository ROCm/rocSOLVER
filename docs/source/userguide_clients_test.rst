
Testing rocSOLVER
==========================

.. toctree::

``rocsolver-test`` executes a suite of `Google tests <https://github.com/google/googletest>`_ (*gtest*) that verifies the correct
functioning of the library; the results computed by rocSOLVER, for random input data, are compared with the results computed by
`NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the CPU.

Calling the rocSOLVER gtest client with the --help flag

.. code-block:: bash

    ./rocsolver-test --help

returns information on different flags that control the behavior of the gtests.
