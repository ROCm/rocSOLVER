/* ************************************************************************
 * Copyright (c) 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_bdsqr.hpp"
#include "testing_gebd2_gebrd.hpp"
#include "testing_gelq2_gelqf.hpp"
#include "testing_geql2_geqlf.hpp"
#include "testing_geqr2_geqrf.hpp"
#include "testing_gesvd.hpp"
#include "testing_getf2_getrf.hpp"
#include "testing_getf2_getrf_npvt.hpp"
#include "testing_getri.hpp"
#include "testing_getrs.hpp"
#include "testing_labrd.hpp"
#include "testing_lacgv.hpp"
#include "testing_larf.hpp"
#include "testing_larfb.hpp"
#include "testing_larfg.hpp"
#include "testing_larft.hpp"
#include "testing_laswp.hpp"
#include "testing_orgbr_ungbr.hpp"
#include "testing_orglx_unglx.hpp"
#include "testing_orgtr_ungtr.hpp"
#include "testing_orgxl_ungxl.hpp"
#include "testing_orgxr_ungxr.hpp"
#include "testing_ormbr_unmbr.hpp"
#include "testing_ormlx_unmlx.hpp"
#include "testing_ormtr_unmtr.hpp"
#include "testing_ormxl_unmxl.hpp"
#include "testing_ormxr_unmxr.hpp"
#include "testing_potf2_potrf.hpp"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[])
try
{
    rocblas_initialize();

    Arguments argus;

    // disable unit_check in client benchmark, it is only
    // used in gtest unit test
    argus.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    argus.timing = 1;

    std::string function;
    char precision;
    rocblas_int device_id;

    // take arguments and set default values
    // (TODO) IMPROVE WORDING/INFORMATION. CHANGE ARGUMENT NAMES FOR
    // MORE RELATED NAMES (THESE ARE BLAS-BASED NAMES)

    // clang-format off
  po::options_description desc("rocsolver client command line options");
  desc.add_options()("help,h", "produces this help message")

        ("sizem,m",
         po::value<rocblas_int>(&argus.M)->default_value(1024),
         "Specific matrix size testing: the number of rows of a matrix.")

        ("sizen,n",
         po::value<rocblas_int>(&argus.N)->default_value(1024),
         "Specific matrix/vector/order size testing: the number of columns of a matrix,"
         "or the order of a system or transformation.")

        ("sizek,k",
         po::value<rocblas_int>(&argus.K)->default_value(1024),
         "Specific...  the number of columns in "
         "A & C  and rows in B.")

        ("size4,S4",
         po::value<rocblas_int>(&argus.S4)->default_value(1024),
         "Extra size value.")

        ("k1",
         po::value<rocblas_int>(&argus.k1)->default_value(1),
         "First index for row interchange, used with laswp. ")

        ("k2",
         po::value<rocblas_int>(&argus.k2)->default_value(2),
         "Last index for row interchange, used with laswp. ")

        ("lda",
         po::value<rocblas_int>(&argus.lda)->default_value(1024),
         "Specific leading dimension of matrix A, is only applicable to "
         "BLAS-2 & BLAS-3: the number of rows.")

        ("ldb",
         po::value<rocblas_int>(&argus.ldb)->default_value(1024),
         "Specific leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3: the number "
         "of rows.")

        ("ldc",
         po::value<rocblas_int>(&argus.ldc)->default_value(1024),
         "Specific leading dimension of matrix C, is only applicable to BLAS-2 & "
         "BLAS-3: the number of rows.")

        ("ldv",
         po::value<rocblas_int>(&argus.ldv)->default_value(1024),
         "Specific leading dimension.")

        ("ldt",
         po::value<rocblas_int>(&argus.ldt)->default_value(1024),
         "Specific leading dimension.")

        ("bsa",
         po::value<rocblas_int>(&argus.bsa)->default_value(1024*1024),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsb",
         po::value<rocblas_int>(&argus.bsb)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsc",
         po::value<rocblas_int>(&argus.bsc)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsp",
         po::value<rocblas_int>(&argus.bsp)->default_value(1024),
         "Specific stride of batched pivots vector Ipiv, is only applicable to batched and strided_batched"
         "factorizations: min(first dimension, second dimension).")

        ("incx",
         po::value<rocblas_int>(&argus.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         po::value<rocblas_int>(&argus.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          po::value<double>(&argus.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         po::value<double>(&argus.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         po::value<std::string>(&function)->default_value("potf2"),
         "LAPACK function to test. Options: potf2, getf2, getrf, getrs")

        ("precision,r",
         po::value<char>(&precision)->default_value('s'), "Options: h,s,d,c,z")

        ("transposeA",
         po::value<char>(&argus.transA_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         po::value<char>(&argus.transB_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeH",
         po::value<char>(&argus.transH_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         po::value<char>(&argus.side_option)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         po::value<char>(&argus.uplo_option)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines")

        ("direct",
         po::value<char>(&argus.direct_option)->default_value('F'),
         "F = forward, B = backward. Only applicable to certain routines")

        ("storev",
         po::value<char>(&argus.storev)->default_value('C'),
         "C = column_wise, R = row_wise. Only applicable to certain routines")

        ("batch",
         po::value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines")

        ("verify,v",
         po::value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         po::value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("perf",
         po::value<rocblas_int>(&argus.perf)->default_value(0),
         "If equal 1, only GPU timing results are collected and printed (default is 0)")

        ("singular",
         po::value<rocblas_int>(&argus.singular)->default_value(0),
         "If equal 1, test with singular matrices (default is 0)")

        ("device",
         po::value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("workmode",
         po::value<char>(&argus.workmode)->default_value('O'),
         "Enables out-of-place computations in some routines")

        ("leftsv",
         po::value<char>(&argus.left_svect)->default_value('N'),
         "Only applicable to certain routines")

        ("rightsv",
         po::value<char>(&argus.right_svect)->default_value('N'),
         "Only applicable to certain routines");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // print help message
    if(vm.count("help"))
    {
        rocblas_cout << desc << std::endl;
        return 0;
    }

    // catch invalid arguments for:

    // precision
    if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
        throw std::invalid_argument("Invalid value for --precision ");

    // deviceID
    if(!argus.perf)
    {
        rocblas_int device_count = query_device_property();
        if(device_count <= device_id)
            throw std::invalid_argument("Invalid Device ID");
    }
    set_device(device_id);

    // operation transA
    if(argus.transA_option != 'N' && argus.transA_option != 'T' && argus.transA_option != 'C')
        throw std::invalid_argument("Invalid value for --transposeA");

    // operation transB
    if(argus.transB_option != 'N' && argus.transB_option != 'T' && argus.transB_option != 'C')
        throw std::invalid_argument("Invalid value for --transposeB");

    // operation transH
    if(argus.transH_option != 'N' && argus.transH_option != 'T' && argus.transH_option != 'C')
        throw std::invalid_argument("Invalid value for --transposeH");

    // side
    if(argus.side_option != 'L' && argus.side_option != 'R' && argus.side_option != 'B')
        throw std::invalid_argument("Invalid value for --side");

    // uplo
    if(argus.uplo_option != 'U' && argus.uplo_option != 'L' && argus.uplo_option != 'F')
        throw std::invalid_argument("Invalid value for --uplo");

    // direct
    if(argus.direct_option != 'F' && argus.direct_option != 'B')
        throw std::invalid_argument("Invalid value for --direct");

    // storev
    if(argus.storev != 'R' && argus.storev != 'C')
        throw std::invalid_argument("Invalid value for --storev");

    // leftsv
    if(argus.left_svect != 'A' && argus.left_svect != 'S' && argus.left_svect != 'O'
       && argus.left_svect != 'N')
        throw std::invalid_argument("Invalid value for --leftsv");

    // rightsv
    if(argus.right_svect != 'A' && argus.right_svect != 'S' && argus.right_svect != 'O'
       && argus.right_svect != 'N')
        throw std::invalid_argument("Invalid value for --rightsv");

    // rightsv
    if(argus.workmode != 'O' && argus.workmode != 'I')
        throw std::invalid_argument("Invalid value for --workmode");

    // select and dispatch function test/benchmark
    // (TODO) MOVE THIS TO A SEPARATE IMPROVED DISPATCH FUNCTION
    if(function == "potf2")
    {
        if(precision == 's')
            testing_potf2_potrf<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "potf2_batched")
    {
        if(precision == 's')
            testing_potf2_potrf<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "potf2_strided_batched")
    {
        if(precision == 's')
            testing_potf2_potrf<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "potrf")
    {
        if(precision == 's')
            testing_potf2_potrf<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "potrf_batched")
    {
        if(precision == 's')
            testing_potf2_potrf<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "potrf_strided_batched")
    {
        if(precision == 's')
            testing_potf2_potrf<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_potf2_potrf<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_potf2_potrf<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_potf2_potrf<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getf2_npvt")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getf2_npvt_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getf2_npvt_strided_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getrf_npvt")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getrf_npvt_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getrf_npvt_strided_batched")
    {
        if(precision == 's')
            testing_getf2_getrf_npvt<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf_npvt<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf_npvt<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf_npvt<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getf2")
    {
        if(precision == 's')
            testing_getf2_getrf<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getf2_batched")
    {
        if(precision == 's')
            testing_getf2_getrf<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getf2_strided_batched")
    {
        if(precision == 's')
            testing_getf2_getrf<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "getrf")
    {
        if(precision == 's')
            testing_getf2_getrf<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getrf_batched")
    {
        if(precision == 's')
            testing_getf2_getrf<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getrf_strided_batched")
    {
        if(precision == 's')
            testing_getf2_getrf<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_getf2_getrf<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_getf2_getrf<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getf2_getrf<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqr2")
    {
        if(precision == 's')
            testing_geqr2_geqrf<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geqr2_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geqr2_strided_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geqrf")
    {
        if(precision == 's')
            testing_geqr2_geqrf<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqrf_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqrf_strided_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqrf_ptr_batched")
    {
        if(precision == 's')
            testing_geqr2_geqrf<true, false, 1, float>(argus);
        else if(precision == 'd')
            testing_geqr2_geqrf<true, false, 1, double>(argus);
        else if(precision == 'c')
            testing_geqr2_geqrf<true, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geqr2_geqrf<true, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geql2")
    {
        if(precision == 's')
            testing_geql2_geqlf<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geql2_batched")
    {
        if(precision == 's')
            testing_geql2_geqlf<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geql2_strided_batched")
    {
        if(precision == 's')
            testing_geql2_geqlf<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "geqlf")
    {
        if(precision == 's')
            testing_geql2_geqlf<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqlf_batched")
    {
        if(precision == 's')
            testing_geql2_geqlf<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "geqlf_strided_batched")
    {
        if(precision == 's')
            testing_geql2_geqlf<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_geql2_geqlf<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_geql2_geqlf<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_geql2_geqlf<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "gelq2")
    {
        if(precision == 's')
            testing_gelq2_gelqf<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gelq2_batched")
    {
        if(precision == 's')
            testing_gelq2_gelqf<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gelq2_strided_batched")
    {
        if(precision == 's')
            testing_gelq2_gelqf<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gelqf")
    {
        if(precision == 's')
            testing_gelq2_gelqf<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "gelqf_batched")
    {
        if(precision == 's')
            testing_gelq2_gelqf<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "gelqf_strided_batched")
    {
        if(precision == 's')
            testing_gelq2_gelqf<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_gelq2_gelqf<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_gelq2_gelqf<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gelq2_gelqf<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "getrs")
    {
        if(precision == 's')
            testing_getrs<false, false, float>(argus);
        else if(precision == 'd')
            testing_getrs<false, false, double>(argus);
        else if(precision == 'c')
            testing_getrs<false, false, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getrs<false, false, rocblas_double_complex>(argus);
    }
    else if(function == "getrs_batched")
    {
        if(precision == 's')
            testing_getrs<true, true, float>(argus);
        else if(precision == 'd')
            testing_getrs<true, true, double>(argus);
        else if(precision == 'c')
            testing_getrs<true, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getrs<true, true, rocblas_double_complex>(argus);
    }
    else if(function == "getrs_strided_batched")
    {
        if(precision == 's')
            testing_getrs<false, true, float>(argus);
        else if(precision == 'd')
            testing_getrs<false, true, double>(argus);
        else if(precision == 'c')
            testing_getrs<false, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getrs<false, true, rocblas_double_complex>(argus);
    }
    else if(function == "gesvd")
    {
        if(precision == 's')
            testing_gesvd<false, false, float>(argus);
        else if(precision == 'd')
            testing_gesvd<false, false, double>(argus);
        else if(precision == 'c')
            testing_gesvd<false, false, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gesvd<false, false, rocblas_double_complex>(argus);
    }
    else if(function == "gesvd_batched")
    {
        if(precision == 's')
            testing_gesvd<true, true, float>(argus);
        else if(precision == 'd')
            testing_gesvd<true, true, double>(argus);
        else if(precision == 'c')
            testing_gesvd<true, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gesvd<true, true, rocblas_double_complex>(argus);
    }
    else if(function == "gesvd_strided_batched")
    {
        if(precision == 's')
            testing_gesvd<false, true, float>(argus);
        else if(precision == 'd')
            testing_gesvd<false, true, double>(argus);
        else if(precision == 'c')
            testing_gesvd<false, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gesvd<false, true, rocblas_double_complex>(argus);
    }
    else if(function == "getri")
    {
        if(precision == 's')
            testing_getri<false, false, float>(argus);
        else if(precision == 'd')
            testing_getri<false, false, double>(argus);
        else if(precision == 'c')
            testing_getri<false, false, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getri<false, false, rocblas_double_complex>(argus);
    }
    else if(function == "getri_batched")
    {
        if(precision == 's')
            testing_getri<true, true, float>(argus);
        else if(precision == 'd')
            testing_getri<true, true, double>(argus);
        else if(precision == 'c')
            testing_getri<true, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getri<true, true, rocblas_double_complex>(argus);
    }
    else if(function == "getri_strided_batched")
    {
        if(precision == 's')
            testing_getri<false, true, float>(argus);
        else if(precision == 'd')
            testing_getri<false, true, double>(argus);
        else if(precision == 'c')
            testing_getri<false, true, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getri<false, true, rocblas_double_complex>(argus);
    }
    else if(function == "getri_outofplace_batched")
    {
        if(precision == 's')
            testing_getri<true, false, float>(argus);
        else if(precision == 'd')
            testing_getri<true, false, double>(argus);
        else if(precision == 'c')
            testing_getri<true, false, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_getri<true, false, rocblas_double_complex>(argus);
    }
    else if(function == "gebd2")
    {
        if(precision == 's')
            testing_gebd2_gebrd<false, false, 0, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<false, false, 0, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<false, false, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<false, false, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gebd2_batched")
    {
        if(precision == 's')
            testing_gebd2_gebrd<true, true, 0, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<true, true, 0, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<true, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<true, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gebd2_strided_batched")
    {
        if(precision == 's')
            testing_gebd2_gebrd<false, true, 0, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<false, true, 0, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<false, true, 0, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<false, true, 0, rocblas_double_complex>(argus);
    }
    else if(function == "gebrd")
    {
        if(precision == 's')
            testing_gebd2_gebrd<false, false, 1, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<false, false, 1, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<false, false, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<false, false, 1, rocblas_double_complex>(argus);
    }
    else if(function == "gebrd_batched")
    {
        if(precision == 's')
            testing_gebd2_gebrd<true, true, 1, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<true, true, 1, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<true, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<true, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "gebrd_strided_batched")
    {
        if(precision == 's')
            testing_gebd2_gebrd<false, true, 1, float>(argus);
        else if(precision == 'd')
            testing_gebd2_gebrd<false, true, 1, double>(argus);
        else if(precision == 'c')
            testing_gebd2_gebrd<false, true, 1, rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_gebd2_gebrd<false, true, 1, rocblas_double_complex>(argus);
    }
    else if(function == "lacgv")
    {
        if(precision == 'c')
            testing_lacgv<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_lacgv<rocblas_double_complex>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "laswp")
    {
        if(precision == 's')
            testing_laswp<float>(argus);
        else if(precision == 'd')
            testing_laswp<double>(argus);
        else if(precision == 'c')
            testing_laswp<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_laswp<rocblas_double_complex>(argus);
    }
    else if(function == "larfg")
    {
        if(precision == 's')
            testing_larfg<float>(argus);
        else if(precision == 'd')
            testing_larfg<double>(argus);
        else if(precision == 'c')
            testing_larfg<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_larfg<rocblas_double_complex>(argus);
    }
    else if(function == "larf")
    {
        if(precision == 's')
            testing_larf<float>(argus);
        else if(precision == 'd')
            testing_larf<double>(argus);
        else if(precision == 'c')
            testing_larf<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_larf<rocblas_double_complex>(argus);
    }
    else if(function == "larft")
    {
        if(precision == 's')
            testing_larft<float>(argus);
        else if(precision == 'd')
            testing_larft<double>(argus);
        else if(precision == 'c')
            testing_larft<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_larft<rocblas_double_complex>(argus);
    }
    else if(function == "larfb")
    {
        if(precision == 's')
            testing_larfb<float>(argus);
        else if(precision == 'd')
            testing_larfb<double>(argus);
        else if(precision == 'c')
            testing_larfb<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_larfb<rocblas_double_complex>(argus);
    }
    else if(function == "labrd")
    {
        if(precision == 's')
            testing_labrd<float>(argus);
        else if(precision == 'd')
            testing_labrd<double>(argus);
        else if(precision == 'c')
            testing_labrd<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_labrd<rocblas_double_complex>(argus);
    }
    else if(function == "org2r")
    {
        if(precision == 's')
            testing_orgxr_ungxr<float, 0>(argus);
        else if(precision == 'd')
            testing_orgxr_ungxr<double, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ung2r")
    {
        if(precision == 'c')
            testing_orgxr_ungxr<rocblas_float_complex, 0>(argus);
        else if(precision == 'z')
            testing_orgxr_ungxr<rocblas_double_complex, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orgqr")
    {
        if(precision == 's')
            testing_orgxr_ungxr<float, 1>(argus);
        else if(precision == 'd')
            testing_orgxr_ungxr<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ungqr")
    {
        if(precision == 'c')
            testing_orgxr_ungxr<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_orgxr_ungxr<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orgl2")
    {
        if(precision == 's')
            testing_orglx_unglx<float, 0>(argus);
        else if(precision == 'd')
            testing_orglx_unglx<double, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ungl2")
    {
        if(precision == 'c')
            testing_orglx_unglx<rocblas_float_complex, 0>(argus);
        else if(precision == 'z')
            testing_orglx_unglx<rocblas_double_complex, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orglq")
    {
        if(precision == 's')
            testing_orglx_unglx<float, 1>(argus);
        else if(precision == 'd')
            testing_orglx_unglx<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unglq")
    {
        if(precision == 'c')
            testing_orglx_unglx<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_orglx_unglx<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orgql")
    {
        if(precision == 's')
            testing_orgxl_ungxl<float, 1>(argus);
        else if(precision == 'd')
            testing_orgxl_ungxl<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ungql")
    {
        if(precision == 'c')
            testing_orgxl_ungxl<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_orgxl_ungxl<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orgbr")
    {
        if(precision == 's')
            testing_orgbr_ungbr<float>(argus);
        else if(precision == 'd')
            testing_orgbr_ungbr<double>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ungbr")
    {
        if(precision == 'c')
            testing_orgbr_ungbr<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_orgbr_ungbr<rocblas_double_complex>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orgtr")
    {
        if(precision == 's')
            testing_orgtr_ungtr<float>(argus);
        else if(precision == 'd')
            testing_orgtr_ungtr<double>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ungtr")
    {
        if(precision == 'c')
            testing_orgtr_ungtr<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_orgtr_ungtr<rocblas_double_complex>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orm2r")
    {
        if(precision == 's')
            testing_ormxr_unmxr<float, 0>(argus);
        else if(precision == 'd')
            testing_ormxr_unmxr<double, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unm2r")
    {
        if(precision == 'c')
            testing_ormxr_unmxr<rocblas_float_complex, 0>(argus);
        else if(precision == 'z')
            testing_ormxr_unmxr<rocblas_double_complex, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ormqr")
    {
        if(precision == 's')
            testing_ormxr_unmxr<float, 1>(argus);
        else if(precision == 'd')
            testing_ormxr_unmxr<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unmqr")
    {
        if(precision == 'c')
            testing_ormxr_unmxr<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_ormxr_unmxr<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orml2")
    {
        if(precision == 's')
            testing_ormlx_unmlx<float, 0>(argus);
        else if(precision == 'd')
            testing_ormlx_unmlx<double, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unml2")
    {
        if(precision == 'c')
            testing_ormlx_unmlx<rocblas_float_complex, 0>(argus);
        else if(precision == 'z')
            testing_ormlx_unmlx<rocblas_double_complex, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ormlq")
    {
        if(precision == 's')
            testing_ormlx_unmlx<float, 1>(argus);
        else if(precision == 'd')
            testing_ormlx_unmlx<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unmlq")
    {
        if(precision == 'c')
            testing_ormlx_unmlx<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_ormlx_unmlx<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "orm2l")
    {
        if(precision == 's')
            testing_ormxl_unmxl<float, 0>(argus);
        else if(precision == 'd')
            testing_ormxl_unmxl<double, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unm2l")
    {
        if(precision == 'c')
            testing_ormxl_unmxl<rocblas_float_complex, 0>(argus);
        else if(precision == 'z')
            testing_ormxl_unmxl<rocblas_double_complex, 0>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ormql")
    {
        if(precision == 's')
            testing_ormxl_unmxl<float, 1>(argus);
        else if(precision == 'd')
            testing_ormxl_unmxl<double, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unmql")
    {
        if(precision == 'c')
            testing_ormxl_unmxl<rocblas_float_complex, 1>(argus);
        else if(precision == 'z')
            testing_ormxl_unmxl<rocblas_double_complex, 1>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ormbr")
    {
        if(precision == 's')
            testing_ormbr_unmbr<float>(argus);
        else if(precision == 'd')
            testing_ormbr_unmbr<double>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unmbr")
    {
        if(precision == 'c')
            testing_ormbr_unmbr<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_ormbr_unmbr<rocblas_double_complex>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "ormtr")
    {
        if(precision == 's')
            testing_ormtr_unmtr<float>(argus);
        else if(precision == 'd')
            testing_ormtr_unmtr<double>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "unmtr")
    {
        if(precision == 'c')
            testing_ormtr_unmtr<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_ormtr_unmtr<rocblas_double_complex>(argus);
        else
            throw std::invalid_argument("This function does not support the given --precision");
    }
    else if(function == "bdsqr")
    {
        if(precision == 's')
            testing_bdsqr<float>(argus);
        else if(precision == 'd')
            testing_bdsqr<double>(argus);
        else if(precision == 'c')
            testing_bdsqr<rocblas_float_complex>(argus);
        else if(precision == 'z')
            testing_bdsqr<rocblas_double_complex>(argus);
    }
    else
        throw std::invalid_argument("Invalid value for --function");

    return 0;
}

catch(const std::invalid_argument& exp)
{
    rocblas_cerr << exp.what() << std::endl;
    return -1;
}
