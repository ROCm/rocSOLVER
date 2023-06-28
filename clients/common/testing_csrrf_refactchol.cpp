/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_refactchol.hpp>

#define TESTING_CSRRF_REFACTCHOL(...) template void testing_csrrf_refactchol<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_REFACTCHOL, FOREACH_REAL_TYPE, APPLY_STAMP)
