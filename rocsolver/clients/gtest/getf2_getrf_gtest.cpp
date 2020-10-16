/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getf2_getrf.hpp"
#include "testing_getf2_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda, singular}
// if singular = 1, then the used matrix for the tests is singular

// case when m = n = 0 will also execute the bad arguments test
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
    {70, 100, 0}};

const vector<int> n_size_range = {
    // quick return
    0,
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    40,
    100,
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 0},
    {640, 640, 1},
    {1000, 1024, 0},
};

const vector<int> large_n_size_range = {
    45, 64, 520, 1024, 2000,
};

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int n_size = std::get<1>(tup);

    Arguments arg;

    arg.M = matrix_size[0];
    arg.N = n_size;
    arg.lda = matrix_size[1];

    arg.timing = 0;
    arg.singular = matrix_size[2];

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = min(arg.M, arg.N);
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class GETF2 : public ::TestWithParam<getrf_tuple>
{
protected:
    GETF2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETRF : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETF2_NPVT : public ::TestWithParam<getrf_tuple>
{
protected:
    GETF2_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class GETRF_NPVT : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF_NPVT() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests
TEST_P(GETF2_NPVT, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 0, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 0, float>(arg);
}

TEST_P(GETF2_NPVT, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 0, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 0, double>(arg);
}

TEST_P(GETF2_NPVT, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 0, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2_NPVT, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 0, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF_NPVT, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 1, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 1, float>(arg);
}

TEST_P(GETRF_NPVT, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 1, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 1, double>(arg);
}

TEST_P(GETRF_NPVT, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 1, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF_NPVT, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, false, 1, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, false, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, false, 1, rocblas_double_complex>(arg);
}

TEST_P(GETF2, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 0, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 0, float>(arg);
}

TEST_P(GETF2, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 0, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 0, double>(arg);
}

TEST_P(GETF2, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 0, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 0, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF, __float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 1, float>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 1, float>(arg);
}

TEST_P(GETRF, __double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 1, double>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 1, double>(arg);
}

TEST_P(GETRF, __float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 1, rocblas_float_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF, __double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, false, 1, rocblas_double_complex>();

    arg.batch_count = 1;
    if(arg.singular == 1)
        testing_getf2_getrf<false, false, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, false, 1, rocblas_double_complex>(arg);
}

// batched tests
TEST_P(GETF2_NPVT, batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 0, float>(arg);
}

TEST_P(GETF2_NPVT, batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 0, double>(arg);
}

TEST_P(GETF2_NPVT, batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2_NPVT, batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF_NPVT, batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 1, float>(arg);
}

TEST_P(GETRF_NPVT, batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 1, double>(arg);
}

TEST_P(GETRF_NPVT, batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF_NPVT, batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<true, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<true, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<true, true, 1, rocblas_double_complex>(arg);
}

TEST_P(GETF2, batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 0, float>(arg);
}

TEST_P(GETF2, batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 0, double>(arg);
}

TEST_P(GETF2, batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2, batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF, batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 1, float>(arg);
}

TEST_P(GETRF, batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 1, double>(arg);
}

TEST_P(GETRF, batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF, batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<true, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<true, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<true, true, 1, rocblas_double_complex>(arg);
}

// strided_batched cases
TEST_P(GETF2_NPVT, strided_batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 0, float>(arg);
}

TEST_P(GETF2_NPVT, strided_batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 0, double>(arg);
}

TEST_P(GETF2_NPVT, strided_batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2_NPVT, strided_batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF_NPVT, strided_batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 1, float>(arg);
}

TEST_P(GETRF_NPVT, strided_batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 1, double>(arg);
}

TEST_P(GETRF_NPVT, strided_batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF_NPVT, strided_batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_npvt_bad_arg<false, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf_npvt<false, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf_npvt<false, true, 1, rocblas_double_complex>(arg);
}

TEST_P(GETF2, strided_batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 0, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 0, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 0, float>(arg);
}

TEST_P(GETF2, strided_batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 0, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 0, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 0, double>(arg);
}

TEST_P(GETF2, strided_batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 0, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 0, rocblas_float_complex>(arg);
}

TEST_P(GETF2, strided_batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 0, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 0, rocblas_double_complex>(arg);
}

TEST_P(GETRF, strided_batched__float)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 1, float>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 1, float>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 1, float>(arg);
}

TEST_P(GETRF, strided_batched__double)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 1, double>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 1, double>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 1, double>(arg);
}

TEST_P(GETRF, strided_batched__float_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 1, rocblas_float_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 1, rocblas_float_complex>(arg);
}

TEST_P(GETRF, strided_batched__double_complex)
{
    Arguments arg = getrf_setup_arguments(GetParam());

    if(arg.M == 0 && arg.N == 0)
        testing_getf2_getrf_bad_arg<false, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    if(arg.singular == 1)
        testing_getf2_getrf<false, true, 1, rocblas_double_complex>(arg);

    arg.singular = 0;
    testing_getf2_getrf<false, true, 1, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETF2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETF2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
