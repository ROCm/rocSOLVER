/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "helpers.h"

template <> float machine_precision() { return static_cast<float>(1.19e-07); }

template <> double machine_precision() { return static_cast<double>(2.22e-16); }
