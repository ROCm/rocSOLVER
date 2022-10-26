/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getri_npvt_outofplace.hpp>

#define TESTING_GETRI_NPVT_OUTOFPLACE(...) \
    template void testing_getri_npvt_outofplace<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRI_NPVT_OUTOFPLACE, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
