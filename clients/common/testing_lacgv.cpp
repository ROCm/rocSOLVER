/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_lacgv.hpp>

#define TESTING_LACGV(...) template void testing_lacgv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LACGV, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
