/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getri.hpp"
#include "testing_getri_npvt.hpp"
#include "testing_getri_npvt_outofplace.hpp"
#include "testing_getri_outofplace.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> getri_tuple;

// each matrix_size_range vector is a {n, lda/ldc, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // invalid
    {-1, 1, 0},
    {20, 5, 0},
    // normal (valid) samples
    {32, 32, 0},
    {50, 50, 1},
    {70, 100, 0},
    {100, 150, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{192, 192, 1}, {500, 600, 1}, {640, 640, 0}, {1000, 1024, 0}, {1200, 1230, 0}};

Arguments getri_setup_arguments(getri_tuple tup, bool outofplace)
{
    Arguments arg;

    arg.set<rocblas_int>("n", tup[0]);
    arg.set<rocblas_int>("lda", tup[1]);

    if(outofplace)
        arg.set<rocblas_int>("ldc", tup[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = tup[2];

    return arg;
}

class GETRI : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getri_setup_arguments(GetParam(), false);

        if(arg.peek<rocblas_int>("n") == 0)
            testing_getri_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getri<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_getri<BATCHED, STRIDED, T>(arg);
    }
};

class GETRI_NPVT : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getri_setup_arguments(GetParam(), false);

        if(arg.peek<rocblas_int>("n") == 0)
            testing_getri_npvt_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getri_npvt<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_getri_npvt<BATCHED, STRIDED, T>(arg);
    }
};

class GETRI_OUTOFPLACE : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI_OUTOFPLACE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getri_setup_arguments(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0)
            testing_getri_outofplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getri_outofplace<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_getri_outofplace<BATCHED, STRIDED, T>(arg);
    }
};

class GETRI_NPVT_OUTOFPLACE : public ::TestWithParam<getri_tuple>
{
protected:
    GETRI_NPVT_OUTOFPLACE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getri_setup_arguments(GetParam(), true);

        if(arg.peek<rocblas_int>("n") == 0)
            testing_getri_npvt_outofplace_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_getri_npvt_outofplace<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_getri_npvt_outofplace<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GETRI, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRI, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRI, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRI, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRI_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRI_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRI_OUTOFPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRI_OUTOFPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRI_OUTOFPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRI_OUTOFPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GETRI, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRI, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRI, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRI, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRI_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRI_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRI_OUTOFPLACE, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRI_OUTOFPLACE, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRI_OUTOFPLACE, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRI_OUTOFPLACE, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GETRI, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRI, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRI, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRI, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRI_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRI_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRI_OUTOFPLACE, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRI_OUTOFPLACE, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRI_OUTOFPLACE, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRI_OUTOFPLACE, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GETRI_NPVT_OUTOFPLACE, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI_NPVT, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI_NPVT, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI_OUTOFPLACE, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI_OUTOFPLACE, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GETRI_NPVT_OUTOFPLACE, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GETRI_NPVT_OUTOFPLACE, ValuesIn(matrix_size_range));
