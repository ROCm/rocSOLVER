.. _clients_label:

*********
Clients
*********

rocSOLVER has an infrastructure for testing and benchmarking similar to that of
`rocBLAS <https://rocblas.readthedocs.io/en/latest/clients.html>`_.

If the clients are built, the client binaries ``rocsolver-test`` and ``rocsolver-bench``
will normally be located in the directory ``<rocsolverDIR>/build/clients/staging``.

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top


Testing rocSOLVER
==========================

``rocsolver-test`` executes a suite of `Google tests <https://github.com/google/googletest>`_ (*gtest*) that verifies the correct
functioning of the library. The results computed by rocSOLVER, given random input data, are compared with the results computed by
`NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the CPU.

Calling the rocSOLVER gtest client with the --help flag

.. code-block:: bash

    ./rocsolver-test --help

returns information on different flags that control the behavior of the gtests.


Benchmarking rocSOLVER
==================================

``rocsolver-bench`` runs any rocSOLVER function with random data of the specified dimensions. It provides basic
performance information (i.e. execution times), and can compare the computed results with
`NETLib LAPACK <https://www.netlib.org/lapack/>`_ on the CPU.

Calling the rocSOLVER bench client with the --help flag

.. code-block:: bash

    ./rocsolver-bench --help

returns information on the different parameters and flags that control the behavior of the benchmark client.


