/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution
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
    rocsparse_solve_policy solve_policy;
    rocsparse_analysis_policy analysis_policy;
};
