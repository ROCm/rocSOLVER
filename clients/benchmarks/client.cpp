/* ************************************************************************
 * Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblascommon/program_options.hpp"
#include "rocsolver_dispatcher.hpp"

int main(int argc, char* argv[])
try
{
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
    options_description desc("rocsolver client command line options");
    desc.add_options()("help,h", "produces this help message")

        ("sizem,m",
         value<rocblas_int>(&argus.M)->default_value(1024),
         "Specific matrix size testing: the number of rows of a matrix.")

        ("sizen,n",
         value<rocblas_int>(&argus.N)->default_value(1024),
         "Specific matrix/vector/order size testing: the number of columns of a matrix,"
         "or the order of a system or transformation.")

        ("sizek,k",
         value<rocblas_int>(&argus.K)->default_value(1024),
         "Specific...  the number of columns in "
         "A & C  and rows in B.")

        ("size4,S4",
         value<rocblas_int>(&argus.S4)->default_value(1024),
         "Extra size value.")

        ("k1",
         value<rocblas_int>(&argus.k1)->default_value(1),
         "First index for row interchange, used with laswp. ")

        ("k2",
         value<rocblas_int>(&argus.k2)->default_value(2),
         "Last index for row interchange, used with laswp. ")

        ("lda",
         value<rocblas_int>(&argus.lda)->default_value(1024),
         "Specific leading dimension of matrix A, is only applicable to "
         "BLAS-2 & BLAS-3: the number of rows.")

        ("ldb",
         value<rocblas_int>(&argus.ldb)->default_value(1024),
         "Specific leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3: the number "
         "of rows.")

        ("ldc",
         value<rocblas_int>(&argus.ldc)->default_value(1024),
         "Specific leading dimension of matrix C, is only applicable to BLAS-2 & "
         "BLAS-3: the number of rows.")

        ("ldv",
         value<rocblas_int>(&argus.ldv)->default_value(1024),
         "Specific leading dimension.")

        ("ldt",
         value<rocblas_int>(&argus.ldt)->default_value(1024),
         "Specific leading dimension.")

        ("bsa",
         value<rocblas_int>(&argus.bsa)->default_value(1024*1024),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsb",
         value<rocblas_int>(&argus.bsb)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsc",
         value<rocblas_int>(&argus.bsc)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsp",
         value<rocblas_int>(&argus.bsp)->default_value(1024),
         "Specific stride of batched pivots vector Ipiv, is only applicable to batched and strided_batched"
         "factorizations: min(first dimension, second dimension).")

        ("bs5",
         value<rocblas_int>(&argus.bs5)->default_value(1024),
         "Specific stride of batched pivots vector Ipiv, is only applicable to batched and strided_batched")

        ("incx",
         value<rocblas_int>(&argus.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<rocblas_int>(&argus.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
         value<double>(&argus.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<double>(&argus.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function)->default_value("potf2"),
         "LAPACK function to test. Options: potf2, getf2, getrf, getrs")

        ("precision,r",
         value<char>(&precision)->default_value('s'), "Options: h,s,d,c,z")

        ("transposeA",
         value<char>(&argus.transA_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&argus.transB_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeH",
         value<char>(&argus.transH_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&argus.side_option)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&argus.uplo_option)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines")

        ("direct",
         value<char>(&argus.direct_option)->default_value('F'),
         "F = forward, B = backward. Only applicable to certain routines")

        ("storev",
         value<char>(&argus.storev)->default_value('C'),
         "C = column_wise, R = row_wise. Only applicable to certain routines")

        ("batch",
         value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines")

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("perf",
         value<rocblas_int>(&argus.perf)->default_value(0),
         "If equal 1, only GPU timing results are collected and printed (default is 0)")

        ("singular",
         value<rocblas_int>(&argus.singular)->default_value(0),
         "If equal 1, test with singular matrices (default is 0)")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("workmode",
         value<char>(&argus.workmode)->default_value('O'),
         "Enables out-of-place computations in some routines")

        ("leftsv",
         value<char>(&argus.left_svect)->default_value('N'),
         "Only applicable to certain routines")

        ("rightsv",
         value<char>(&argus.right_svect)->default_value('N'),
         "Only applicable to certain routines")

        ("evect",
         value<char>(&argus.evect)->default_value('N'),
         "Only applicable to certain routines")

        ("itype",
         value<char>(&argus.itype)->default_value('1'),
         "Only applicable to certain routines");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    // print help message
    if(vm.count("help"))
    {
        rocsolver_cout << desc << std::endl;
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

    // evect
    if(argus.evect != 'V' && argus.evect != 'I' && argus.evect != 'N')
        throw std::invalid_argument("Invalid value for --evect");

    // workmode
    if(argus.workmode != 'O' && argus.workmode != 'I')
        throw std::invalid_argument("Invalid value for --workmode");

    // itype
    if(argus.itype != '1' && argus.itype != '2' && argus.itype != '3')
        throw std::invalid_argument("Invalid value for --itype");

    // select and dispatch function test/benchmark
    rocsolver_dispatcher::invoke(function, precision, argus);

    return 0;
}

catch(const std::invalid_argument& exp)
{
    rocsolver_cerr << exp.what() << std::endl;
    return -1;
}
