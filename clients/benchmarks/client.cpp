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

        // test options
        ("function,f",
         value<std::string>(&function)->default_value("potf2"),
         "The LAPACK function to test.")

        ("precision,r",
         value<char>(&precision)->default_value('s'),
         "Precision of the LAPACK function to test. Options: h, s, d, c, z.")

        ("batch_count",
         value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines.")

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No).")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop (default: 10).")

        ("perf",
         value<rocblas_int>(&argus.perf)->default_value(0),
         "Ignore CPU timing results? 0 = No, 1 = Yes (default: No).")

        ("singular",
         value<rocblas_int>(&argus.singular)->default_value(0),
         "Test with degenerate matrices? 0 = No, 1 = Yes (default: No).")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs.")

        // size options
        ("m",
         value<rocblas_int>()->default_value(128),
         "Matrix size parameter. Typically, the number of rows of a matrix.")

        ("n",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. Typically, the number of columns of a matrix,"
         "or the order of a system or transformation.")

        ("nrhs",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. Typically, the number of columns of a matrix"
         "on the right-hand side.")

        ("k",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. Typically, the number of rows and columns to be reduced.")

        ("nc",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. The number of columns of matrix C. Only applicable to bdsqr.")

        ("nu",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. The number of columns of matrix U. Only applicable to bdsqr.")

        ("nv",
         value<rocblas_int>()->default_value(128),
         "Matrix/vector size parameter. The number of columns of matrix V. Only applicable to bdsqr.")

        // leading dimension options
        ("lda",
         value<rocblas_int>(),
         "Leading dimension of matrices A.")

        ("ldb",
         value<rocblas_int>(),
         "Leading dimension of matrices B.")

        ("ldc",
         value<rocblas_int>(),
         "Leading dimension of matrices C.")

        ("ldt",
         value<rocblas_int>(),
         "Leading dimension of matrices T.")

        ("ldu",
         value<rocblas_int>(),
         "Leading dimension of matrices U.")

        ("ldv",
         value<rocblas_int>(),
         "Leading dimension of matrices V.")

        ("ldw",
         value<rocblas_int>(),
         "Leading dimension of matrices W.")

        ("ldx",
         value<rocblas_int>(),
         "Leading dimension of matrices X.")

        ("ldy",
         value<rocblas_int>(),
         "Leading dimension of matrices Y.")

        // stride options
        ("strideA",
         value<rocblas_stride>(),
         "Stride for matrices/vectors A.")

        ("strideB",
         value<rocblas_stride>(),
         "Stride for matrices/vectors B.")

        ("bsc",
         value<rocblas_int>(&argus.bsc)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("strideD",
         value<rocblas_stride>(),
         "Stride for matrices/vectors D.")

        ("strideE",
         value<rocblas_stride>(),
         "Stride for matrices/vectors E.")

        ("strideQ",
         value<rocblas_stride>(),
         "Stride for vectors tau and ipiv.")

        ("strideP",
         value<rocblas_stride>(),
         "Stride for vectors tau and ipiv.")

        ("strideS",
         value<rocblas_stride>(),
         "Stride for matrices/vectors S.")

        ("strideU",
         value<rocblas_stride>(),
         "Stride for matrices/vectors U.")

        ("strideV",
         value<rocblas_stride>(),
         "Stride for matrices/vectors V.")

        // increment options
        ("incx",
         value<rocblas_int>()->default_value(1),
         "Increment between values in vector x.")

        ("incy",
         value<rocblas_int>(&argus.incy)->default_value(1),
         "increment between values in y vector")

        // coefficient options
        ("alpha",
         value<double>(&argus.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<double>(&argus.beta)->default_value(0.0), "specifies the scalar beta")

        // transpose options
        ("trans",
         value<char>()->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose.")

        ("transposeB",
         value<char>(&argus.transB_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose.")

        ("transposeH",
         value<char>(&argus.transH_option)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose.")

        // other options
        ("k1",
         value<rocblas_int>()->default_value(1),
         "First index for row interchange. Only applicable to laswp.")

        ("k2",
         value<rocblas_int>()->default_value(2),
         "Last index for row interchange. Only applicable to laswp.")

        ("side",
         value<char>()->default_value('L'),
         "L = left, R = right. Only applicable to certain routines.")

        ("uplo",
         value<char>()->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines.")

        ("direct",
         value<char>()->default_value('F'),
         "F = forward, B = backward. Only applicable to certain routines.")

        ("storev",
         value<char>()->default_value('C'),
         "C = column_wise, R = row_wise. Only applicable to certain routines.")

        ("fast_alg",
         value<char>()->default_value('O'),
         "Enables out-of-place computations. Only applicable to gesvd.")

        ("left_svect",
         value<char>()->default_value('N'),
         "Computation type for left singular vectors. Only applicable to gesvd.")

        ("right_svect",
         value<char>()->default_value('N'),
         "Computation type for right singular vectors. Only applicable to gesvd.")

        ("evect",
         value<char>()->default_value('N'),
         "Computation type for eigenvectors. Only applicable to certain routines.")

        ("jobz",
         value<char>()->default_value('N'),
         "Computation type for eigenvectors. Only applicable to certain routines.")

        ("itype",
         value<char>()->default_value('1'),
         "Problem type for generalized eigenproblems. Only applicable to certain routines.");
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

    argus.populate(vm);

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

    argus.validate_operation("trans");

    // operation transB
    if(argus.transB_option != 'N' && argus.transB_option != 'T' && argus.transB_option != 'C')
        throw std::invalid_argument("Invalid value for --transposeB");

    // operation transH
    if(argus.transH_option != 'N' && argus.transH_option != 'T' && argus.transH_option != 'C')
        throw std::invalid_argument("Invalid value for --transposeH");

    argus.validate_side("side");
    argus.validate_fill("uplo");
    argus.validate_direct("direct");
    argus.validate_storev("storev");
    argus.validate_svect("left_svect");
    argus.validate_svect("right_svect");
    argus.validate_workmode("fast_alg");
    argus.validate_evect("evect");
    argus.validate_evect("jobz");
    argus.validate_itype("itype");

    // select and dispatch function test/benchmark
    rocsolver_dispatcher::invoke(function, precision, argus);

    return 0;
}

catch(const std::invalid_argument& exp)
{
    rocsolver_cerr << exp.what() << std::endl;
    return -1;
}
