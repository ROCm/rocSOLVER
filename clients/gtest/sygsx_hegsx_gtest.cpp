/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sygsx_hegsx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> sygst_tuple;

// each matrix_size_range is a {n, lda, ldb}

// each type_range is a {itype, uplo}

// case when n = 0, itype = 1, and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> type_range = {{'1', 'L'}, {'2', 'L'}, {'1', 'U'}, {'3', 'U'}};

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
    {152, 152, 152},
    {640, 640, 640},
    {1000, 1024, 1024},
};

Arguments sygst_setup_arguments(sygst_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> type = std::get<1>(tup);

    Arguments arg;

    arg.N = matrix_size[0];
    arg.lda = matrix_size[1];
    arg.ldb = matrix_size[2];

    arg.itype = type[0];
    arg.uplo_option = type[1];

    arg.timing = 0;

    // only testing standard use case for strides
    // strides are ignored in normal and batched tests
    arg.bsa = arg.lda * arg.N;
    arg.bsb = arg.ldb * arg.N;

    return arg;
}

template <bool BLOCKED>
class SYGSX_HEGSX : public ::TestWithParam<sygst_tuple>
{
protected:
    SYGSX_HEGSX() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void test_fixture()
    {
        Arguments arg = sygst_setup_arguments(GetParam());

        if(arg.itype == '1' && arg.uplo_option == 'U' && arg.N == 0)
            testing_sygsx_hegsx_bad_arg<BATCHED, STRIDED, BLOCKED, float>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_sygsx_hegsx<BATCHED, STRIDED, BLOCKED, float>(arg);
    }
};

class SYGS2 : public SYGSX_HEGSX<false>
{
};

class HEGS2 : public SYGSX_HEGSX<false>
{
};

class SYGST : public SYGSX_HEGSX<true>
{
};

class HEGST : public SYGSX_HEGSX<true>
{
};

// non-batch tests

TEST_P(SYGS2, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(SYGS2, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(HEGS2, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(HEGS2, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

TEST_P(SYGST, __float)
{
    test_fixture<false, false, float>();
}

TEST_P(SYGST, __double)
{
    test_fixture<false, false, double>();
}

TEST_P(HEGST, __float_complex)
{
    test_fixture<false, false, rocblas_float_complex>();
}

TEST_P(HEGST, __double_complex)
{
    test_fixture<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYGS2, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(SYGS2, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(HEGS2, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(HEGS2, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

TEST_P(SYGST, batched__float)
{
    test_fixture<true, true, float>();
}

TEST_P(SYGST, batched__double)
{
    test_fixture<true, true, double>();
}

TEST_P(HEGST, batched__float_complex)
{
    test_fixture<true, true, rocblas_float_complex>();
}

TEST_P(HEGST, batched__double_complex)
{
    test_fixture<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYGS2, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(SYGS2, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(HEGS2, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(HEGS2, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

TEST_P(SYGST, strided_batched__float)
{
    test_fixture<false, true, float>();
}

TEST_P(SYGST, strided_batched__double)
{
    test_fixture<false, true, double>();
}

TEST_P(HEGST, strided_batched__float_complex)
{
    test_fixture<false, true, rocblas_float_complex>();
}

TEST_P(HEGST, strided_batched__double_complex)
{
    test_fixture<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGS2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGS2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGS2,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGS2,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGST,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGST,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
