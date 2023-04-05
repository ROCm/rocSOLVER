/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_splitlu.hpp>

#define TESTING_CSRRF_SPLITLU(...) template void testing_csrrf_splitlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_SPLITLU, FOREACH_REAL_TYPE, APPLY_STAMP)
