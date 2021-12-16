.. _tuning_label:

*******************************
Tuning rocSOLVER Performance
*******************************

Some constant parameters in rocSOLVER can be tuned to affect the performance of the
library functions in a given context (like for a particular matrix size and/or shape, for example).
A description of these tunable constants is presented in this section.

To facilitate the description, the constants are grouped by the function or familly of functions they affect.
Some aspects of the involved algorithms are also depicted here for the sake of clarity, however,
this section if not intended to be a review of the well known methods for the different matrix computations.
These constants are specific to the rocSOLVER implementation of the algorithms, and are only
described within this context.

All the described constants are gathered in the header file
``<rocsolverDIR>/library/src/include/ideal_sizes.hpp``, and
for any change to take effect, the library must be :ref:`rebuilt from source <userguide_install>`;
these are not run-time arguments of the associated API functions.

.. warning::
    The effect that changing the values of the tunable constants
    will have on the performance of the library is hard to predict, and this analysis is out
    of the scope of this document. Advanced users and developers trying to tune these values
    should proceed with caution; there is no guarantee that new values will improve (or worsen)
    the performance of the associated functions.

.. toctree::
   :maxdepth: 4

.. contents:: Table of contents
   :local:
   :backlinks: top



geqr2/geqrf and geql2/geqlf functions
======================================

GEQxF_GEQx2_SWITCHSIZE
-----------------------

GEQxF_GEQx2_BLOCKSIZE
----------------------


gerq2/gerqf and gelq2/gelqf functions
========================================

GExQF_GExQ2_SWITCHSIZE
-----------------------

GExQF_GExQ2_BLOCKSIZE
----------------------


org2r/orgqr, orgl2/orglq and org2l/orgql functions
====================================================

ORGxx_UNGxx_SWITCHSIZE
-----------------------

ORGxx_UNGxx_BLOCKSIZE
----------------------


orm2r/ormqr, orml2/ormlq and orm2l/ormql functions
=======================================================

ORMxx_UNMxx_SWITCHSIZE
-----------------------

ORMxx_UNMxx_BLOCKSIZE
----------------------


getf2/getrf functions
========================

GETF2_MAX_COLS
---------------

GETF2_MAX_THDS
---------------

GETF2_OPTIM_NGRP
-----------------

GETRF_NUM_INTERVALS
--------------------

GETRF_INTERVALS
----------------

GETRF_BLKSIZES
---------------

GETRF_BATCH_NUM_INTERVALS
----------------------------

GETRF_BATCH_INTERVALS
----------------------

GETRF_BATCH_BLKSIZES
-------------------------

GETRF_NPVT_NUM_INTERVALS
--------------------------

GETRF_NPVT_INTERVALS
----------------------

GETRF_NPVT_BLKSIZES
---------------------

GETRF_NPVT_BATCH_NUM_INTERVALS
-------------------------------

GETRF_NPVT_BATCH_INTERVALS
---------------------------

GETRF_NPVT_BATCH_BLKSIZES
---------------------------


potf2/potrf functions
=========================

POTRF_POTF2_SWITCHSIZE
------------------------


getri function
================

GETRI_MAX_COLS
---------------

GETRI_TINY_SIZE
----------------

GETRI_NUM_INTERVALS
--------------------

GETRI_INTERVALS
----------------

GETRI_BLKSIZES
----------------

GETRI_BATCH_TINY_SIZE
-----------------------

GETRI_BATCH_NUM_INTERVALS
--------------------------

GETRI_BATCH_INTERVALS
------------------------

GETRI_BATCH_BLKSIZES
---------------------


trtri function
=================

TRTRI_MAX_COLS
---------------

TRTRI_NUM_INTERVALS
--------------------

TRTRI_INTERVALS
----------------

TRTRI_BLKSIZES
---------------

TRTRI_BATCH_NUM_INTERVALS
--------------------------

TRTRI_BATCH_INTERVALS
----------------------

TRTRI_BATCH_BLKSIZES
---------------------


gebd2/gebrd functions
=========================

GEBRD_GEBD2_SWITCHSIZE
-----------------------


sytd2/sytrd and hetd2/hetrd functions
==========================================

xxTRD_xxTD2_BLOCKSIZE
----------------------

xxTRD_xxTD2_SWITCHSIZE
-----------------------


sygs2/sygst and hegs2/hegst functions
======================================

xxGST_xxGS2_BLOCKSIZE
------------------------


gesvd function
==================

THIN_SVD_SWITCH
------------------


stedc function
===================

STEDC_MIN_DC_SIZE
-------------------


sytf2/sytrf functions
=======================

SYTRF_BLOCKSIZE
----------------



