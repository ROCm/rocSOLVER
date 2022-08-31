/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_larf.hpp>

#define TESTING_LARF(...) template void testing_larf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LARF, FOREACH_SCALAR_TYPE, APPLY_STAMP)
