/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_labrd.hpp>

#define TESTING_LABRD(...) template void testing_labrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LABRD, FOREACH_SCALAR_TYPE, APPLY_STAMP)
