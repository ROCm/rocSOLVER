/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_refactlu.hpp>

#define TESTING_CSRRF_REFACTLU(...) template void testing_csrrf_refactlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_REFACTLU, FOREACH_REAL_TYPE, APPLY_STAMP)
