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

    // ------------------------
    // used in LU factorization
    // ------------------------
    rocsparse_mat_descr descrT;
    rocsparse_mat_descr descrL;
    rocsparse_mat_descr descrU;

    rocsparse_mat_info infoT;
    rocsparse_mat_info infoL;
    rocsparse_mat_info infoU;

    // -------------------------------
    // used in  Cholesky factorization
    // -------------------------------
    rocsparse_mat_descr descrTchol;
    rocsparse_mat_descr descrLchol;

    rocsparse_mat_info infoTchol;
    rocsparse_mat_info infoLchol;
    rocsparse_mat_info infoLtchol;

    rocsparse_solve_policy solve_policy;
    rocsparse_analysis_policy analysis_policy;

    rocsolver_rfinfo_mode mode;
    bool analyzed;
};
