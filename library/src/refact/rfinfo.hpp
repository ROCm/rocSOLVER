/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

#include "hip_check.h"
#include "rocsparse_check.h"
#include "rocblas_check.h"

struct rocsolver_rfinfo_
{
    rocsparse_handle sphandle = nullptr;
    rocsparse_mat_descr descrL = nullptr;
    rocsparse_mat_descr descrU = nullptr;
    rocsparse_mat_descr descrT = nullptr;
    rocsparse_mat_info infoL = nullptr;
    rocsparse_mat_info infoU = nullptr;
    rocsparse_mat_info infoT = nullptr;

    // constructor
    rocsolver_rfinfo_(rocblas_handle handle)
    {

         
        // create sparse handle
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_handle(&sphandle));

        // use handle->stream to sphandle->stream
        hipStream_t stream;
        THROW_IF_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        THROW_IF_HIP_ERROR(rocsparse_set_stream(sphandle, stream));

        // create and set matrix descriptors
        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descrL));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(descrL, rocsparse_matrix_type_triangular));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_diag_type(descrL, rocsparse_diag_type_unit));

        THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descrU));
        THROW_IF_ROCSPARSE_ERROR(rocsparse_set_mat_type(descrU, rocsparse_matrix_type_triangular));
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

    // acts like destructor but can throw exceptions
    void destroy() 
    {
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_handle(sphandle));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_descr(descrL));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_descr(descrU));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_descr(descrT));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_info(infoL));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_info(infoU));
        THROW_IF_ROCSPARSE_ERROR( rocsparse_destroy_mat_info(infoT));

        sphandle = nullptr;
        descrL = nullptr;
        descrU = nullptr;
        descrT = nullptr;
        infoL = nullptr;
        infoU = nullptr;
        infoT = nullptr;
    }
    

    // destructor should not throw exceptions
    ~rocsolver_rfinfo_()
    {
        if (sphandle != nullptr) { rocsparse_destroy_handle(sphandle); };
        if (descrL != nullptr) { rocsparse_destroy_mat_descr(descrL); }; 
        if (descrU != nullptr) { rocsparse_destroy_mat_descr(descrU); };
        if (descrT != nullptr) { rocsparse_destroy_mat_descr(descrT); };
        if (infoL != nullptr) { rocsparse_destroy_mat_info(infoL); };
        if (infoU != nullptr) { rocsparse_destroy_mat_info(infoU); };
        if (infoT != nullptr) { rocsparse_destroy_mat_info(infoT); };
    }
};
