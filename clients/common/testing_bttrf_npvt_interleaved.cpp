/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_bttrf_npvt_interleaved.hpp>

#define TESTING_BTTRF_NPVT_INTERLEAVED(...) \
    template void testing_bttrf_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_BTTRF_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
