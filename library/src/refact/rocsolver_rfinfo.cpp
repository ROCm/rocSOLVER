/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <new>

#ifdef HAVE_ROCSPARSE
#include "rocsolver_rfinfo.hpp"
#endif

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

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

#define GOTO_IF_ROCSPARSE_ERROR(fcn, result, error_label) \
    do                                                    \
    {                                                     \
        rocsparse_status _status = (fcn);                 \
        if(_status != rocsparse_status_success)           \
        {                                                 \
            result = rocsparse2rocblas_status(_status);   \
            goto error_label;                             \
        }                                                 \
    } while(0)

extern "C" rocblas_status rocsolver_create_rfinfo(rocsolver_rfinfo* rfinfo, rocblas_handle handle)
{
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return rocblas_status_invalid_handle;

    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    rocsolver_rfinfo_* impl = new(std::nothrow) rocsolver_rfinfo_{};
    if(!impl)
        return rocblas_status_memory_error;

    rocblas_status result;

    // create sparse handle
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_handle(&impl->sphandle), result, cleanup);

    // use handle->stream to sphandle->stream
    hipStream_t stream;
    GOTO_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_stream(impl->sphandle, stream), result, cleanup);

    // create and set matrix descriptors
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&impl->descrL), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(impl->descrL, rocsparse_matrix_type_general),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(impl->descrL, rocsparse_index_base_zero),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(impl->descrL, rocsparse_fill_mode_lower),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(impl->descrL, rocsparse_diag_type_unit),
                            result, cleanup);

    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&impl->descrU), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(impl->descrU, rocsparse_matrix_type_general),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(impl->descrU, rocsparse_index_base_zero),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(impl->descrU, rocsparse_fill_mode_upper),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(impl->descrU, rocsparse_diag_type_non_unit),
                            result, cleanup);

    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&impl->descrT), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(impl->descrT, rocsparse_matrix_type_general),
                            result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(impl->descrT, rocsparse_index_base_zero),
                            result, cleanup);

    // create info holders
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&impl->infoL), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&impl->infoU), result, cleanup);
    GOTO_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&impl->infoT), result, cleanup);

    impl->solve_policy = rocsparse_solve_policy_auto;
    impl->analysis_policy = rocsparse_analysis_policy_reuse;
    *rfinfo = impl;
    return rocblas_status_success;
cleanup:
    rocsparse_destroy_mat_info(impl->infoT);
    rocsparse_destroy_mat_info(impl->infoU);
    rocsparse_destroy_mat_info(impl->infoL);
    rocsparse_destroy_mat_descr(impl->descrT);
    rocsparse_destroy_mat_descr(impl->descrU);
    rocsparse_destroy_mat_descr(impl->descrL);
    rocsparse_destroy_handle(impl->sphandle);
    delete impl;
    return result;
#else
    return rocblas_status_not_implemented;
#endif
}

extern "C" rocblas_status rocsolver_destroy_rfinfo(rocsolver_rfinfo rfinfo)
{
#ifdef HAVE_ROCSPARSE
    if(!rfinfo)
        return rocblas_status_invalid_pointer;

    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(rfinfo->infoT));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(rfinfo->infoU));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(rfinfo->infoL));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(rfinfo->descrT));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(rfinfo->descrU));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(rfinfo->descrL));
    ROCSPARSE_CHECK(rocsparse_destroy_handle(rfinfo->sphandle));
    delete rfinfo;

    return rocblas_status_success;
#else
    return rocblas_status_not_implemented;
#endif
}
