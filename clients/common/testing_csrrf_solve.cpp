/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_solve.hpp>

#define TESTING_CSRRF_SOLVE(...) template void testing_csrrf_solve<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_SOLVE, FOREACH_REAL_TYPE, APPLY_STAMP)
