/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cstdio>
#include <iostream>
#include <rocblas.h>
#include <string>
#include <type_traits>
#include <vector>

#include "rocblas_test.hpp"

/*!\file
 * \brief provide common utilities
 */

// We use rocsolver_cout and rocsolver_cerr instead of std::cout, std::cerr, stdout
// and stderr, for thread-safe IO.
//
// All stdio and std::ostream functions related to stdout and stderr are
// poisoned, as are functions which can create buffer overflows, or which are
// inherently thread-unsafe.
//
// This must come after the header #includes above, to avoid poisoning system
// headers.
//
// This is only enabled for rocblas-test and rocblas-bench.
//
// If you are here because of a poisoned identifier error, here is the rationale
// for each included identifier:
//
// cout, stdout: rocsolver_cout should be used instead, for thread-safe and atomic
// line buffering cerr, stderr: rocsolver_cerr should be used instead, for
// thread-safe and atomic line buffering clog: C++ stream which should not be
// used gets: Always unsafe; buffer-overflows; removed from later versions of
// the language; use fgets puts, putchar, fputs, printf, fprintf, vprintf,
// vfprintf: Use rocsolver_cout or rocsolver_cerr sprintf, vsprintf: Possible buffer
// overflows; us snprintf or vsnprintf instead strerror: Thread-unsafe; use
// snprintf / dprintf with %m or strerror_* alternatives strtok: Thread-unsafe;
// use strtok_r gmtime, ctime, asctime, localtime: Thread-unsafe tmpnam:
// Thread-unsafe; use mkstemp or related functions instead putenv: Use setenv
// instead clearenv, fcloseall, ecvt, fcvt: Miscellaneous thread-unsafe
// functions sleep: Might interact with signals by using alarm(); use
// nanosleep() instead

#if defined(GOOGLE_TEST) || defined(ROCBLAS_BENCH)
#undef stdout
#undef stderr
#pragma GCC poison cout cerr clog stdout stderr gets puts putchar fputs fprintf printf sprintf    \
    vfprintf vprintf vsprintf perror strerror strtok gmtime ctime asctime localtime tmpnam putenv \
        clearenv fcloseall ecvt fcvt sleep
#define BOOST_ASSERT_MSG_OSTREAM rocsolver_cerr
// Suppress warnings about hipMalloc(), hipFree() except in rocblas-test and
// rocblas-bench
#undef hipMalloc
#undef hipFree
#endif

static constexpr char LIMITED_MEMORY_STRING[]
    = "Error: Attempting to allocate more memory than available.";

// TODO: This is dependent on internal gtest behaviour.
// Compared with result.message() when a test ended. Note that "Succeeded\n" is
// added to the beginning of the message automatically by gtest, so this must be
// compared.
static constexpr char LIMITED_MEMORY_STRING_GTEST[]
    = "Succeeded\nError: Attempting to allocate more memory than available.";

/* ============================================================================================
 */
/*! \brief  local handle which is automatically created and destroyed  */
class rocblas_local_handle
{
    rocblas_handle m_handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&m_handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(m_handle);
    }

    rocblas_local_handle(const rocblas_local_handle&) = delete;
    rocblas_local_handle(rocblas_local_handle&&) = delete;
    rocblas_local_handle& operator=(const rocblas_local_handle&) = delete;
    rocblas_local_handle& operator=(rocblas_local_handle&&) = delete;

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return m_handle;
    }
    operator const rocblas_handle&() const
    {
        return m_handle;
    }
};

/* ============================================================================================
 */
/*  device query and print out their ID and name */
rocblas_int query_device_property();

/*  set current device to device_id */
void set_device(rocblas_int device_id);

/* ============================================================================================
 */

template <typename T>
void print_strided_batched(const char* name,
                           T* A,
                           rocblas_int n1,
                           rocblas_int n2,
                           rocblas_int n3,
                           rocblas_int s1,
                           rocblas_int s2,
                           rocblas_int s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    rocsolver_cout << "---------- " << name << " ----------\n";
    int max_size = 8;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                rocsolver_cout << A[(i1 * s1) + (i2 * s2) + (i3 * s3)] << "|";
            }
            rocsolver_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            rocsolver_cout << "\n";
    }
    rocsolver_cout << std::flush;
}
