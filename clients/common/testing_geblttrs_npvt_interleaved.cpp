/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_geblttrs_npvt_interleaved.hpp>

#define TESTING_GEBLTTRS_NPVT_INTERLEAVED(...) \
    template void testing_geblttrs_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEBLTTRS_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
