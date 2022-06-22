/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sygvj_hegvj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<printable_char>> sygvj_tuple;

// each matrix_size_range is a {n, lda, ldb, singular}
// if singular = 1, then the used matrix for the tests is not positive definite

// each type_range is a {itype, evect, uplo}

// case when n = 0, itype = 1, evect = 'N', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<printable_char>> type_range
    = {{'1', 'N', 'U'}, {'2', 'N', 'L'}, {'3', 'N', 'U'},
       {'1', 'V', 'L'}, {'2', 'V', 'U'}, {'3', 'V', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {20, 5, 5, 0},
    // normal (valid) samples
    {20, 30, 20, 1},
    {35, 35, 35, 0},
    {50, 50, 60, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192, 192, 0},
    {256, 270, 256, 0},
    {300, 300, 310, 0},
};

Arguments sygvj_setup_arguments(sygvj_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<printable_char> type = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("evect", type[1]);
    arg.set<char>("uplo", type[2]);

    arg.set<double>("abstol", 0);
    arg.set<rocblas_int>("max_sweeps", 100);

    // only testing standard use case/defaults for strides

    arg.timing = 0;
    arg.singular = matrix_size[3];

    return arg;
}

class SYGVJ_HEGVJ : public ::TestWithParam<sygvj_tuple>
{
protected:
    SYGVJ_HEGVJ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygvj_setup_arguments(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("evect") == 'N'
           && arg.peek<char>("uplo") == 'U' && arg.peek<rocblas_int>("n") == 0)
            testing_sygvj_hegvj_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        if(arg.singular == 1)
            testing_sygvj_hegvj<BATCHED, STRIDED, T>(arg);

        arg.singular = 0;
        testing_sygvj_hegvj<BATCHED, STRIDED, T>(arg);
    }
};

class SYGVJ : public SYGVJ_HEGVJ
{
};

class HEGVJ : public SYGVJ_HEGVJ
{
};

// non-batch tests

TEST_P(SYGVJ, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVJ, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVJ, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVJ, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(SYGVJ, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(SYGVJ, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(HEGVJ, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(HEGVJ, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched cases

TEST_P(SYGVJ, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYGVJ, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEGVJ, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEGVJ, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYGVJ,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVJ,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HEGVJ,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVJ,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
