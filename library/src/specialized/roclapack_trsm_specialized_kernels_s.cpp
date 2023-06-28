/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution
 * ************************************************************************ */

#include "roclapack_trsm_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRSM_MEM(0, 0, float);
INSTANTIATE_TRSM_LOWER(0, 0, float, float*);
INSTANTIATE_TRSM_UPPER(0, 0, float, float*);

INSTANTIATE_TRSM_MEM(0, 1, float);
INSTANTIATE_TRSM_LOWER(0, 1, float, float*);
INSTANTIATE_TRSM_UPPER(0, 1, float, float*);

INSTANTIATE_TRSM_MEM(1, 0, float);
INSTANTIATE_TRSM_LOWER(1, 0, float, float* const*);
INSTANTIATE_TRSM_UPPER(1, 0, float, float* const*);
