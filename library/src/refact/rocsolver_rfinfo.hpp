/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#ifdef ROCSOLVER_WITH_ROCSPARSE
#include "rocsparse.hpp"

struct rocsolver_rfinfo_
{
    rocsparse_handle sphandle = nullptr;
    rocsparse_mat_descr descrL = nullptr;
    rocsparse_mat_descr descrU = nullptr;
    rocsparse_mat_descr descrT = nullptr;
    rocsparse_mat_info infoL = nullptr;
    rocsparse_mat_info infoU = nullptr;
    rocsparse_mat_info infoT = nullptr;

    rocsparse_solve_policy solve_policy = rocsparse_solve_policy_auto;

    // rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_force;
    rocsparse_analysis_policy analysis_policy = rocsparse_analysis_policy_reuse;

    // constructor
    rocsolver_rfinfo_(rocblas_handle handle)
    {
        // create sparse handle
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_handle(&sphandle));

        // use handle->stream to sphandle->stream
        hipStream_t stream;
        THROW_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_stream(sphandle, stream));

        // ----------------------------------------------------------
        // TODO: check whether to use triangular type or general type
        // ----------------------------------------------------------
        // rocsparse_matrix_type L_type = rocsparse_matrix_type_triangular;
        // rocsparse_matrix_type U_type = rocsparse_matrix_type_triangular;
        rocsparse_matrix_type const L_type = rocsparse_matrix_type_general;
        rocsparse_matrix_type const U_type = rocsparse_matrix_type_general;

        // create and set matrix descriptors
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descrL));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(descrL, L_type));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descrL, rocsparse_diag_type_unit));

        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descrU));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(descrU, U_type));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descrU, rocsparse_diag_type_non_unit));

        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descrT));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(descrT, rocsparse_matrix_type_general));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrT, rocsparse_index_base_zero));

        // create info holders
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&infoL));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&infoU));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&infoT));
    }

    // acts like destructor but can return status
    rocblas_status destroy()
    {
        ROCSPARSE_CHECK(rocsparse_destroy_handle(sphandle));
        sphandle = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrL));
        descrL = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrU));
        descrU = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrT));
        descrT = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_info(infoL));
        infoL = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_info(infoU));
        infoU = nullptr;
        ROCSPARSE_CHECK(rocsparse_destroy_mat_info(infoT));
        infoT = nullptr;

        return rocblas_status_success;
    }

    // destructor
    ~rocsolver_rfinfo_()
    {
        if(sphandle != nullptr)
        {
            rocsparse_destroy_handle(sphandle);
            sphandle = nullptr;
        };
        if(descrL != nullptr)
        {
            rocsparse_destroy_mat_descr(descrL);
            descrL = nullptr;
        };
        if(descrU != nullptr)
        {
            rocsparse_destroy_mat_descr(descrU);
            descrU = nullptr;
        };
        if(descrT != nullptr)
        {
            rocsparse_destroy_mat_descr(descrT);
            descrT = nullptr;
        };
        if(infoL != nullptr)
        {
            rocsparse_destroy_mat_info(infoL);
            infoL = nullptr;
        };
        if(infoU != nullptr)
        {
            rocsparse_destroy_mat_info(infoU);
            infoU = nullptr;
        };
        if(infoT != nullptr)
        {
            rocsparse_destroy_mat_info(infoT);
            infoT = nullptr;
        };
    }
};
typedef struct rocsolver_rfinfo_* rocsolver_rfinfo;

#endif
