/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sygv_hegv.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> sygv_tuple;

// each matrix_size_range is a {n, lda, ldb}

// each type_range is a {itype, jobz, uplo}

// case when n = 0, itype = 1, jobz = 'N', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> type_range = {{'1', 'N', 'L'}, {'2', 'V', 'L'}, {'3', 'N', 'L'},
                                         {'1', 'V', 'U'}, {'2', 'N', 'U'}, {'3', 'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {20, 5, 5},
    // normal (valid) samples
    {50, 50, 50},
    {70, 100, 110},
    {130, 130, 130}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192},
    {256, 270, 280},
    {300, 300, 300},
};

Arguments sygv_setup_arguments(sygv_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> type = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];
    arg.ldb = matrix_size[2];

    arg.itype = type[0];
    arg.evect = type[1];
    arg.uplo_option = type[2];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;
    arg.bsb = arg.ldb * arg.N;
    arg.bsp = arg.N;

    return arg;
}

class SYGV : public ::TestWithParam<sygv_tuple>
{
protected:
    SYGV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class HEGV : public ::TestWithParam<sygv_tuple>
{
protected:
    HEGV() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(SYGV, __float)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, false, float>();

    arg.batch_count = 1;
    testing_sygv_hegv<false, false, float>(arg);
}

TEST_P(SYGV, __double)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, false, double>();

    arg.batch_count = 1;
    testing_sygv_hegv<false, false, double>(arg);
}

TEST_P(HEGV, __float_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, false, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_sygv_hegv<false, false, rocblas_float_complex>(arg);
}

TEST_P(HEGV, __double_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, false, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_sygv_hegv<false, false, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(SYGV, batched__float)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<true, true, float>();

    arg.batch_count = 3;
    testing_sygv_hegv<true, true, float>(arg);
}

TEST_P(SYGV, batched__double)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<true, true, double>();

    arg.batch_count = 3;
    testing_sygv_hegv<true, true, double>(arg);
}

TEST_P(HEGV, batched__float_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<true, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sygv_hegv<true, true, rocblas_float_complex>(arg);
}

TEST_P(HEGV, batched__double_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<true, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sygv_hegv<true, true, rocblas_double_complex>(arg);
}

// strided_batched cases

TEST_P(SYGV, strided_batched__float)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, true, float>();

    arg.batch_count = 3;
    testing_sygv_hegv<false, true, float>(arg);
}

TEST_P(SYGV, strided_batched__double)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, true, double>();

    arg.batch_count = 3;
    testing_sygv_hegv<false, true, double>(arg);
}

TEST_P(HEGV, strided_batched__float_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, true, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sygv_hegv<false, true, rocblas_float_complex>(arg);
}

TEST_P(HEGV, strided_batched__double_complex)
{
    Arguments arg = sygv_setup_arguments(GetParam());

    if(arg.itype == '1' && arg.evect == 'N' && arg.uplo_option == 'U' && arg.N == 0)
        testing_sygv_hegv_bad_arg<false, true, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sygv_hegv<false, true, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGV,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGV,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGV,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGV,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
