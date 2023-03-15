/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <new>

#include "rfinfo.hpp"

extern "C" rocblas_status rocsolver_create_rfinfo(rocsolver_rfinfo* rfinfo, rocblas_handle handle)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    try
    {
        *rfinfo = new rocsolver_rfinfo_(handle);
    } catch (const std::bad_alloc&) {
        return rocblas_status_memory_error;
    }

    return rocblas_status_success;
}

extern "C" rocblas_status rocsolver_destroy_rfinfo(rocsolver_rfinfo rfinfo)
{
    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    delete rfinfo;

    return rocblas_status_success;
}
