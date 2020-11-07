/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sytxx_hetxx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> sytrd_tuple;

// each matrix_size_range is a {n, lda}

// case when n = 0 and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

Arguments sytrd_setup_arguments(sytrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char uplo = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.uplo_option = uplo;

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsp = arg.N;
    arg.bsa = arg.lda * arg.N;

    return arg;
}

class SYTD2 : public ::TestWithParam<sytrd_tuple>
{
protected:
    SYTD2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class SYTRD : public ::TestWithParam<sytrd_tuple>
{
protected:
    SYTRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class HETD2 : public ::TestWithParam<sytrd_tuple>
{
protected:
    HETD2() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class HETRD : public ::TestWithParam<sytrd_tuple>
{
protected:
    HETRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// non-batch tests

TEST_P(SYTD2, __float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 0, float>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 0, float>(arg);
}

TEST_P(SYTD2, __double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 0, double>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 0, double>(arg);
}

TEST_P(HETD2, __float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 0, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 0, rocblas_float_complex>(arg);
}

TEST_P(HETD2, __double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 0, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 0, rocblas_double_complex>(arg);
}

TEST_P(SYTRD, __float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 1, float>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 1, float>(arg);
}

TEST_P(SYTRD, __double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 1, double>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 1, double>(arg);
}

TEST_P(HETRD, __float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 1, rocblas_float_complex>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 1, rocblas_float_complex>(arg);
}

TEST_P(HETRD, __double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, false, 1, rocblas_double_complex>();

    arg.batch_count = 1;
    testing_sytxx_hetxx<false, false, 1, rocblas_double_complex>(arg);
}

// batched tests

TEST_P(SYTD2, batched__float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 0, float>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 0, float>(arg);
}

TEST_P(SYTD2, batched__double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 0, double>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 0, double>(arg);
}

TEST_P(HETD2, batched__float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 0, rocblas_float_complex>(arg);
}

TEST_P(HETD2, batched__double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 0, rocblas_double_complex>(arg);
}

TEST_P(SYTRD, batched__float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 1, float>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 1, float>(arg);
}

TEST_P(SYTRD, batched__double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 1, double>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 1, double>(arg);
}

TEST_P(HETRD, batched__float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 1, rocblas_float_complex>(arg);
}

TEST_P(HETRD, batched__double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<true, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<true, true, 1, rocblas_double_complex>(arg);
}

// strided_batched cases

TEST_P(SYTD2, strided_batched__float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 0, float>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 0, float>(arg);
}

TEST_P(SYTD2, strided_batched__double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 0, double>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 0, double>(arg);
}

TEST_P(HETD2, strided_batched__float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 0, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 0, rocblas_float_complex>(arg);
}

TEST_P(HETD2, strided_batched__double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 0, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 0, rocblas_double_complex>(arg);
}

TEST_P(SYTRD, strided_batched__float)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 1, float>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 1, float>(arg);
}

TEST_P(SYTRD, strided_batched__double)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 1, double>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 1, double>(arg);
}

TEST_P(HETRD, strided_batched__float_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 1, rocblas_float_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 1, rocblas_float_complex>(arg);
}

TEST_P(HETRD, strided_batched__double_complex)
{
    Arguments arg = sytrd_setup_arguments(GetParam());

    if(arg.uplo_option == 'U' && arg.N == 0)
        testing_sytxx_hetxx_bad_arg<false, true, 1, rocblas_double_complex>();

    arg.batch_count = 3;
    testing_sytxx_hetxx<false, true, 1, rocblas_double_complex>(arg);
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTD2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTD2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HETD2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETD2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HETRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
