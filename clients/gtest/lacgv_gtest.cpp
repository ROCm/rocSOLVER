/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 * 1)Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2)Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * ************************************************************************ */

#include "testing_lacgv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> lacgv_tuple;

// each range is a {n,inc}

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {1, 0},
    // normal (valid) samples
    {10, 1},
    {10, -1},
    {20, 2},
    {30, 3},
    {30, -3}};

// for daily_lapack tests
const vector<vector<int>> large_range
    = {{192, 10}, {192, -10}, {250, 20}, {500, 30}, {1500, 40}, {1500, -40}};

Arguments lacgv_setup_arguments(lacgv_tuple tup)
{
    Arguments arg;

    arg.set<rocblas_int>("n", tup[0]);
    arg.set<rocblas_int>("incx", tup[1]);

    return arg;
}

class LACGV : public ::TestWithParam<lacgv_tuple>
{
protected:
    LACGV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = lacgv_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_lacgv_bad_arg<T>();

        testing_lacgv<T>(arg);
    }
};

// non-batch tests

TEST_P(LACGV, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LACGV, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, LACGV, ValuesIn(large_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, LACGV, ValuesIn(range));
