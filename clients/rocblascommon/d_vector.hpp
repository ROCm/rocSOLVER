/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cinttypes>
#include <cstdio>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <rocblas/rocblas.h>

#include "common_host_helpers.hpp"
#include "rocblas_init.hpp"
#include "rocblas_test.hpp"

/* ============================================================================================
 */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T, size_t PAD, typename U>
class d_vector
{
private:
    size_t size, bytes;

public:
    inline size_t nmemb() const noexcept
    {
        return size;
    }

    d_vector(size_t s)
        : size(s)
        , bytes(s ? s * sizeof(T) : sizeof(T))
    {
    }

    T* device_vector_setup()
    {
        T* d;
        if((hipMalloc)(&d, bytes) != hipSuccess)
        {
            fmt::print(stderr, "Error allocating {} bytes ({} GB)\n", bytes, bytes >> 30);
            d = nullptr;
        }
        return d;
    }

    void device_vector_check(T* d) {}

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};
