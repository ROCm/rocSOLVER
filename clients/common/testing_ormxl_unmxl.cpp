/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include <testing_ormxl_unmxl.hpp>

#define TESTING_ORMXL_UNMXL(...) template void testing_ormxl_unmxl<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMXL_UNMXL, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
