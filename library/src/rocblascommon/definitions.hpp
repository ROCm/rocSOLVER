/* ************************************************************************
 * Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once

#include <fmt/core.h>
#include <rocblas/rocblas.h>

/*******************************************************************************
 * Definitions
 ******************************************************************************/
#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                         \
    do                                                                      \
    {                                                                       \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;           \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                              \
        {                                                                   \
            return get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                   \
    } while(0)

#define RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)               \
    do                                                                \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            return TMP_STATUS_FOR_CHECK;                              \
        }                                                             \
    } while(0)

#define THROW_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                         \
    do                                                                     \
    {                                                                      \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;          \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                             \
        {                                                                  \
            throw get_rocblas_status_for_hip_status(TMP_STATUS_FOR_CHECK); \
        }                                                                  \
    } while(0)

#define THROW_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                \
    do                                                                \
    {                                                                 \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)            \
        {                                                             \
            throw TMP_STATUS_FOR_CHECK;                               \
        }                                                             \
    } while(0)

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                                            \
    do                                                                                        \
    {                                                                                         \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                             \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                                \
        {                                                                                     \
            fmt::print(stderr, "hip error code: '{}':{} at {}:{}\n",                          \
                       hipGetErrorName(TMP_STATUS_FOR_CHECK), TMP_STATUS_FOR_CHECK, __FILE__, \
                       __LINE__);                                                             \
        }                                                                                     \
    } while(0)

#define PRINT_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                                       \
    do                                                                                       \
    {                                                                                        \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                        \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                   \
        {                                                                                    \
            fmt::print(stderr, "rocblas error: '{}':{} at {}:{}\n",                          \
                       rocblas_status_to_string(TMP_STATUS_FOR_CHECK), TMP_STATUS_FOR_CHECK, \
                       __FILE__, __LINE__);                                                  \
        }                                                                                    \
    } while(0)

#define PRINT_AND_RETURN_IF_ROCBLAS_ERROR(INPUT_STATUS_FOR_CHECK)                            \
    do                                                                                       \
    {                                                                                        \
        rocblas_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;                        \
        if(TMP_STATUS_FOR_CHECK != rocblas_status_success)                                   \
        {                                                                                    \
            fmt::print(stderr, "rocblas error: '{}':{} at {}:{}\n",                          \
                       rocblas_status_to_string(TMP_STATUS_FOR_CHECK), TMP_STATUS_FOR_CHECK, \
                       __FILE__, __LINE__);                                                  \
            return TMP_STATUS_FOR_CHECK;                                                     \
        }                                                                                    \
    } while(0)
