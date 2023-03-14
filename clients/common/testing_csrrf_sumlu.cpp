/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_sumlu.hpp>

#define TESTING_CSRRF_SUMLU(...) template void testing_csrrf_sumlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_SUMLU, FOREACH_REAL_TYPE, APPLY_STAMP)
