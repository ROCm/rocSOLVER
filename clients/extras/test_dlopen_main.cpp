/* ************************************************************************
 * Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include <dlfcn.h>

#include <gtest/gtest.h>

// Tensorflow uses dlopen to load the ROCm libraries.
// https://github.com/ROCmSoftwarePlatform/rocSOLVER/issues/230
TEST(TestDynamicLinking, AllSymbolsResolved)
{
    ASSERT_NE(dlopen(ROCSOLVER_LIB_NAME, RTLD_NOW | RTLD_LOCAL), nullptr) << dlerror();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
