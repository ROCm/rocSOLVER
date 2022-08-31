/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_ormtr_unmtr.hpp>

#define TESTING_ORMTR_UNMTR(...) template void testing_ormtr_unmtr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMTR_UNMTR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
