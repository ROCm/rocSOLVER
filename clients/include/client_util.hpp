/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// Utility macros for explicit template instantiation with rocBLAS types

#define APPLY_STAMP(STAMP, ...) STAMP(__VA_ARGS__)
#define FOREACH_BOOLEAN_INT(STAMP, F, ...) \
    F(STAMP, ## __VA_ARGS__, 0)               \
    F(STAMP, ## __VA_ARGS__, 1)
#define FOREACH_BOOLEAN_0(STAMP, F, ...) \
    F(STAMP, ## __VA_ARGS__, false)         \
    F(STAMP, ## __VA_ARGS__, true)
#define FOREACH_BOOLEAN_1(STAMP, F, ...) \
    F(STAMP, ## __VA_ARGS__, false)         \
    F(STAMP, ## __VA_ARGS__, true)
#define FOREACH_REAL_TYPE(STAMP, F, ...)       \
    F(STAMP, ## __VA_ARGS__, float)                 \
    F(STAMP, ## __VA_ARGS__, double)                
#define FOREACH_COMPLEX_TYPE(STAMP, F, ...)       \
    F(STAMP, ## __VA_ARGS__, rocblas_float_complex) \
    F(STAMP, ## __VA_ARGS__, rocblas_double_complex)
#define FOREACH_SCALAR_TYPE(STAMP, F, ...)       \
    F(STAMP, ## __VA_ARGS__, float)                 \
    F(STAMP, ## __VA_ARGS__, double)                \
    F(STAMP, ## __VA_ARGS__, rocblas_float_complex) \
    F(STAMP, ## __VA_ARGS__, rocblas_double_complex)
#define FOREACH_MATRIX_DATA_LAYOUT(STAMP, F, ...) \
    F(STAMP, ## __VA_ARGS__, false, false)           \
    F(STAMP, ## __VA_ARGS__, true, false)            \
    F(STAMP, ## __VA_ARGS__, false, true)
#define INSTANTIATE(STAMP, F, ...) F(STAMP, ## __VA_ARGS__)
