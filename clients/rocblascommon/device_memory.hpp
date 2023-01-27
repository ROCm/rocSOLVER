/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <memory>

#include <hip/hip_runtime_api.h>

#include "common_host_helpers.hpp"

struct device_deleter
{
    void operator()(void* p) const
    {
        // Throwing an error when hipFree fails will likely result in throwing
        // from a destructor, which should be avoided. However, we don't really
        // have many options. Worst comes to worst, throwing will result in
        // std::terminate being called, which is perhaps not such a bad thing
        // in the test and bench clients where this is used.
        THROW_IF_HIP_ERROR(hipFree(p));
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter>;
