.. _clients_label:

*********
Clients
*********

.. toctree::
   :maxdepth: 4
   :caption: Contents:

rocSOLVER has a basic/preliminary infrastructure for testing and benchmarking similar to that of rocBLAS.

On a normal installation, client binaries ``rocsolver-test`` and ``rocsolver-bench``
should be located in the directory **<rocsolverDIR>/build/clients/staging**.

Testing rocSOLVER
==========================

``rocsolver-test`` executes a suite of `Google tests <https://github.com/google/googletest>`_ (*gtest*) that verifies the correct
functioning of the library; the results computed by rocSOLVER, for random input data, are compared with the results computed by
`NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the CPU.

Calling the rocSOLVER gtest client with the --help flag

.. code-block:: bash

    ./rocsolver-test --help

returns information on different flags that control the behavior of the gtests.

Benchmarking rocSOLVER
==================================

``rocsolver-bench`` runs any rocSOLVER function with random data of the specified dimensions; it compares the computed results, and provides basic
performance information (as for now, execution times).

Similarly,

.. code-block:: bash

    ./rocsolver-bench --help

returns information on how to use the rocSOLVER benchmark client.

