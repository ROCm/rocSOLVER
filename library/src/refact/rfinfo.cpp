/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rfinfo.hpp"

extern "C" rocblas_status rocsolver_rfinfo_create(rocsolver_rfinfo* rfinfo, rocblas_handle handle)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    *rfinfo = new rocsolver_rfinfo_(handle);

    return rocblas_status_success;
}

extern "C" rocblas_status rocsolver_rfinfo_destroy(rocsolver_rfinfo rfinfo)
{
    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    delete rfinfo;

    return rocblas_status_success;
}
