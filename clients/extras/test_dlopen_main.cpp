/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
