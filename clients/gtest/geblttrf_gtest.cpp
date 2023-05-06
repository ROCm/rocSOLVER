/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_geblttrf_npvt.hpp"
#include "testing_geblttrf_npvt_interleaved.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef vector<int> geblttrf_tuple;

// each matrix_size_range vector is a {nb, nblocks, lda, ldb, ldc, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when nb = 0 and nblocks = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 1, 1, 0},
    {1, 0, 1, 1, 1, 0},
    // invalid
    {-1, 1, 1, 1, 1, 0},
    {1, -1, 1, 1, 1, 0},
    {10, 2, 1, 1, 1, 0},
    // normal (valid) samples
    {32, 1, 32, 32, 32, 0},
    {16, 2, 20, 16, 16, 1},
    {10, 7, 10, 20, 10, 0},
    {10, 10, 10, 10, 20, 1},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {{32, 6, 32, 32, 32, 0},
                                                     {50, 10, 60, 50, 50, 1},
                                                     {32, 10, 32, 40, 32, 0},
                                                     {32, 20, 32, 32, 40, 0}};

Arguments geblttrf_setup_arguments(geblttrf_tuple tup)
{
    Arguments arg;

    arg.set<rocblas_int>("nb", tup[0]);
    arg.set<rocblas_int>("nblocks", tup[1]);
    arg.set<rocblas_int>("lda", tup[2]);
    arg.set<rocblas_int>("ldb", tup[3]);
    arg.set<rocblas_int>("ldc", tup[4]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = tup[5];

    return arg;
}

class GEBLTTRF_NPVT : public ::TestWithParam<geblttrf_tuple>
{
protected:
    GEBLTTRF_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = geblttrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("nblocks") == 0)
            testing_geblttrf_npvt_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_geblttrf_npvt<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_geblttrf_npvt<BATCHED, STRIDED, T>(arg);
    }
};

class GEBLTTRF_NPVT_INTERLEAVED : public ::TestWithParam<geblttrf_tuple>
{
protected:
    GEBLTTRF_NPVT_INTERLEAVED() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = geblttrf_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("nb") == 0 && arg.peek<rocblas_int>("nblocks") == 0)
            testing_geblttrf_npvt_interleaved_bad_arg<T>();

        arg.batch_count = 3;
        if(arg.singular == 1)
            testing_geblttrf_npvt_interleaved<T>(arg);

        arg.singular = 0;
        testing_geblttrf_npvt_interleaved<T>(arg);
    }
};

// non-batch tests

TEST_P(GEBLTTRF_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBLTTRF_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBLTTRF_NPVT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBLTTRF_NPVT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEBLTTRF_NPVT, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEBLTTRF_NPVT, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEBLTTRF_NPVT, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEBLTTRF_NPVT, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEBLTTRF_NPVT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEBLTTRF_NPVT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEBLTTRF_NPVT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEBLTTRF_NPVT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// interleaved_batched tests

TEST_P(GEBLTTRF_NPVT_INTERLEAVED, interleaved_batched__float)
{
    run_tests<float>();
}

TEST_P(GEBLTTRF_NPVT_INTERLEAVED, interleaved_batched__double)
{
    run_tests<double>();
}

TEST_P(GEBLTTRF_NPVT_INTERLEAVED, interleaved_batched__float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(GEBLTTRF_NPVT_INTERLEAVED, interleaved_batched__double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEBLTTRF_NPVT, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEBLTTRF_NPVT, ValuesIn(matrix_size_range));

INSTANTIATE_TEST_SUITE_P(daily_lapack, GEBLTTRF_NPVT_INTERLEAVED, ValuesIn(large_matrix_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GEBLTTRF_NPVT_INTERLEAVED, ValuesIn(matrix_size_range));
