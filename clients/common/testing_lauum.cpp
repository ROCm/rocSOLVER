/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_lauum.hpp>

#define TESTING_LAUUM(...) template void testing_lauum<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LAUUM, FOREACH_SCALAR_TYPE, APPLY_STAMP)
