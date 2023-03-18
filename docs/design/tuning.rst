.. _tuning_label:

*******************************
Tuning rocSOLVER Performance
*******************************

Some compile-time parameters in rocSOLVER can be modified to tune the performance
of the library functions in a given context (e.g., for a particular matrix size or shape).
A description of these tunable constants is presented in this section.

To facilitate the description, the constants are grouped by the family of functions they affect.
Some aspects of the involved algorithms are also depicted here for the sake of clarity; however,
this section is not intended to be a review of the well-known methods for different matrix computations.
These constants are specific to the rocSOLVER implementation and are only described within that context.

All described constants can be found in ``library/src/include/ideal_sizes.hpp``.
These are not run-time arguments for the associated API functions. The library must be
:ref:`rebuilt from source<userguide_install_source>` for any change to take effect.

.. warning::
    The effect of changing a tunable constant on the performance of the library is difficult
    to predict, and such analysis is beyond the scope of this document. Advanced users and
    developers tuning these values should proceed with caution. New values may (or may not)
    improve or worsen the performance of the associated functions.

.. contents:: Table of contents
   :local:
   :backlinks: top


More to come later...



