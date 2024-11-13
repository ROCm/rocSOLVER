/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <new>

#include "rocsolver_spinfo.hpp"

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#ifndef HAVE_ROCSPARSE
#if defined(_WIN32) && !defined(ROCSOLVER_STATIC_LIB)
#include <windows.h>
#elif !defined(ROCSOLVER_STATIC_LIB) /* defined(_WIN32) && !defined(ROCSOLVER_STATIC_LIB) */
#include <dlfcn.h>
#endif /* defined(_WIN32) && !defined(ROCSOLVER_STATIC_LIB)*/
#endif /* HAVE_ROCSPARSE */

#define GOTO_IF_ROCBLAS_ERROR(fcn, result, error_label) \
    do                                                  \
    {                                                   \
        rocblas_status _status = (fcn);                 \
        if(_status != rocblas_status_success)           \
        {                                               \
            result = _status;                           \
            goto error_label;                           \
        }                                               \
    } while(0)

#define GOTO_IF_ROCSPARSE_ERROR(fcn, result, error_label)          \
    do                                                             \
    {                                                              \
        rocsparse_status _status = (fcn);                          \
        if(_status != rocsparse_status_success)                    \
        {                                                          \
            result = rocsolver::rocsparse2rocblas_status(_status); \
            goto error_label;                                      \
        }                                                          \
    } while(0)

ROCSOLVER_BEGIN_NAMESPACE

template <typename Fn>
static bool load_function(void* handle, const char* symbol, Fn& fn)
{
#ifndef ROCSOLVER_STATIC_LIB
#ifdef _WIN32
    fn = (Fn)(GetProcAddress((HMODULE)handle, symbol));
    bool err = !fn;
#else
    fn = (Fn)(dlsym(handle, symbol));
    char* err = dlerror(); // clear errors
#ifndef NDEBUG
    if(err)
        fmt::print(stderr, "rocsolver: error loading {:s}: {:s}\n", symbol, err);
#endif /* NDEBUG */
#endif /* _WIN32 */
    return !err;
#else
    return false;
#endif /* ROCSOLVER_STATIC_LIB */
}

static bool load_rocsparse()
{
#ifndef ROCSOLVER_STATIC_LIB
#ifdef _WIN32
    // Library users will need to call SetErrorMode(SEM_FAILCRITICALERRORS) if
    // they wish to avoid an error message box when this library is not found.
    // The call is not done by rocSOLVER directly, as it is not thread-safe and
    // will affect the global state of the program.
    void* handle = LoadLibraryW(L"rocsparse.dll");
#else
    void* handle = dlopen("librocsparse.so.1", RTLD_NOW | RTLD_LOCAL);
    char* err = dlerror(); // clear errors
#ifndef NDEBUG
    if(!handle)
        fmt::print(stderr, "rocsolver: error loading librocsparse.so.1: {:s}\n", err);
#endif /* NDEBUG */
#endif /* _WIN32 */
    if(!handle)
        return false;
    if(!load_function(handle, "rocsparse_create_handle", g_sparse_create_handle))
        return false;
    if(!load_function(handle, "rocsparse_destroy_handle", g_sparse_destroy_handle))
        return false;

    if(!load_function(handle, "rocsparse_set_stream", g_sparse_set_stream))
        return false;
    if(!load_function(handle, "rocsparse_create_mat_descr", g_sparse_create_mat_descr))
        return false;
    if(!load_function(handle, "rocsparse_destroy_mat_descr", g_sparse_destroy_mat_descr))
        return false;
    if(!load_function(handle, "rocsparse_set_mat_type", g_sparse_set_mat_type))
        return false;
    if(!load_function(handle, "rocsparse_set_mat_index_base", g_sparse_set_mat_index_base))
        return false;
    if(!load_function(handle, "rocsparse_set_mat_fill_mode", g_sparse_set_mat_fill_mode))
        return false;
    if(!load_function(handle, "rocsparse_set_mat_diag_type", g_sparse_set_mat_diag_type))
        return false;
    if(!load_function(handle, "rocsparse_create_mat_info", g_sparse_create_mat_info))
        return false;
    if(!load_function(handle, "rocsparse_destroy_mat_info", g_sparse_destroy_mat_info))
        return false;

    if(!load_function(handle, "rocsparse_scsr2dense", g_sparse_scsr2dense))
        return false;
    if(!load_function(handle, "rocsparse_dcsr2dense", g_sparse_dcsr2dense))
        return false;
    if(!load_function(handle, "rocsparse_ccsr2dense", g_sparse_ccsr2dense))
        return false;
    if(!load_function(handle, "rocsparse_zcsr2dense", g_sparse_zcsr2dense))
        return false;
    return true;
#else /* ROCSOLVER_STATIC_LIB */
    return false;
#endif
}

static bool try_load_rocsparse()
{
    // Function-scope static initialization has been thread-safe since C++11.
    // There is an implicit mutex guarding the initialization.
    static bool result = load_rocsparse();
    return result;
}

ROCSOLVER_END_NAMESPACE

extern "C" rocblas_status rocsolver_create_spinfo(rocsolver_spinfo* spinfo, rocblas_handle handle)
{
#ifndef HAVE_ROCSPARSE
    if(!rocsolver::try_load_rocsparse())
        return rocblas_status_not_implemented;
#endif

    if(!handle)
        return rocblas_status_invalid_handle;

    if(!spinfo)
        return rocblas_status_invalid_pointer;

    auto impl = new(std::nothrow) rocsolver_spinfo_{};
    if(!impl)
        return rocblas_status_memory_error;

    rocblas_status result;

    // create sparse handle
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_handle(&impl->sphandle), result, cleanup);

    // use handle->stream to sphandle->stream
    hipStream_t stream;
    GOTO_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_stream(impl->sphandle, stream), result, cleanup);

    // setup descrA
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(impl->descrA)), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(impl->descrA, rocsparse_matrix_type_general),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(impl->descrA, rocsparse_index_base_zero),
                            result, cleanup);

    *spinfo = impl;
    return rocblas_status_success;
cleanup:
    rocsparse_destroy_handle(impl->sphandle);
    rocsparse_destroy_mat_descr(impl->descrA);

    delete impl;
    return result;
}

extern "C" rocblas_status rocsolver_destroy_spinfo(rocsolver_spinfo spinfo)
{
    if(!spinfo)
        return rocblas_status_invalid_pointer;

    ROCSPARSE_CHECK(rocsparse_destroy_handle(spinfo->sphandle));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(spinfo->descrA));
    delete spinfo;
    return rocblas_status_success;
}
