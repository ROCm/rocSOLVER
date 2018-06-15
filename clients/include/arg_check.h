/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _ARG_CHECK_H
#define _ARG_CHECK_H

#include "rocsolver.h"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/* =====================================================================

    Google Unit check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google
 * Unit check.
 */

/* ========================================Gtest Arg Check
 * ===================================================== */

/*! \brief Template: tests arguments are valid */

void potf2_arg_check(rocsolver_status status, rocsolver_int N);

void getf2_arg_check(rocsolver_status status, rocsolver_int M, rocsolver_int N);

void getrf_arg_check(rocsolver_status status, rocsolver_int M, rocsolver_int N);

template <typename T> void verify_not_nan(T arg);

template <typename T> void verify_equal(T arg1, T arg2, const char *message);

#endif
