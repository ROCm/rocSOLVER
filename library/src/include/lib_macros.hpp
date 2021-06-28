/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

/*
 * ===========================================================================
 *    common location for macros that are used in many rocSOLVER routines
 * ===========================================================================
 */

#ifdef _WIN32
#define ROCSOLVER_KERNEL static __global__
#else
#define ROCSOLVER_KERNEL __global__
#endif
