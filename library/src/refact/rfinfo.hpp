/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsparse.hpp"

struct rocsolver_rfinfo_
{
    rocsparse_handle sphandle;
    rocsparse_mat_descr descrL;
    rocsparse_mat_descr descrU;
    rocsparse_mat_descr descrT;
    rocsparse_mat_info infoL;
    rocsparse_mat_info infoU;
    rocsparse_mat_info infoT;

    // constructor
    rocsolver_rfinfo_(rocblas_handle handle)
    {
        /*        // create sparse handle
        rocsparse_handle sphandle;
        rocsparse_create_handle(&sphandle);

        // use handle->stream to sphandle->stream
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);
        rocsparse_set_stream(sphandle, stream);

        // create and set matrix descriptors
        rocsparse_create_mat_descr(&descrL);
        rocsparse_set_mat_type(descrL, rocsparse_matrix_type_triangular);
        rocsparse_set_mat_index_base(descrL, rocsparse_index_base_zero);
        rocsparse_set_mat_fill_mode(descrL, rocsparse_fill_mode_lower);
        rocsparse_set_mat_diag_type(descrL, rocsparse_diag_type_unit);

        rocsparse_create_mat_descr(&descrU);
        rocsparse_set_mat_type(descrU, rocsparse_matrix_type_triangular);
        rocsparse_set_mat_index_base(descrU, rocsparse_index_base_zero);
        rocsparse_set_mat_fill_mode(descrU, rocsparse_fill_mode_upper);
        rocsparse_set_mat_diag_type(descrU, rocsparse_diag_type_non_unit);

        rocsparse_create_mat_descr(&descrT);
        rocsparse_set_mat_type(descrT, rocsparse_matrix_type_general);
        rocsparse_set_mat_index_base(descrT, rocsparse_index_base_zero);

        // create info holders
        rocsparse_create_mat_info(&infoL);
        rocsparse_create_mat_info(&infoU);
        rocsparse_create_mat_info(&infoT);*/
    }

    // destructor
    ~rocsolver_rfinfo_()
    {
        /*        rocsparse_destroy_handle(sphandle);
        rocsparse_destroy_mat_descr(descrL);
        rocsparse_destroy_mat_descr(descrU);
        rocsparse_destroy_mat_descr(descrT);
        rocsparse_destroy_mat_info(infoL);
        rocsparse_destroy_mat_info(infoU);
        rocsparse_destroy_mat_info(infoT);*/
    }
};
