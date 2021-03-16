/* ************************************************************************
 * Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblascommon/program_options.hpp"
#include "rocsolver_dispatcher.hpp"

using namespace boost;

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
    // clang-format off
    options_description desc("rocsolver client command line options");
    desc.add_options()("help,h", "Produces this help message.")

        // test options
        ("batch_count",
         value<rocblas_int>(&argus.batch_count)->default_value(1),
            "Number of matrices or problem instances in the batch.\n"
            "                           Only applicable to batch routines.\n"
            "                           ")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
            "Set the default device to be used for subsequent program runs.\n"
            "                           ")

        ("function,f",
         value<std::string>(&function)->default_value("potf2"),
            "The LAPACK function to test.\n"
            "                           Options are: getf2, getrf, gesvd_batched, etc.\n"
            "                           ")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
            "Iterations to run inside the GPU timing loop.\n"
            "                           Reported time will be the average.\n"
            "                           ")

        ("perf",
         value<rocblas_int>(&argus.perf)->default_value(0),
            "Ignore CPU timing results? 0 = No, 1 = Yes.\n"
            "                           This forces the client to print only the GPU time and the error if requested.\n"
            "                           ")

        ("precision,r",
         value<char>(&precision)->default_value('s'),
            "Precision to be used in the tests.\n"
            "                           Options are: s, d, c, z.\n"
            "                           ")

        ("singular",
         value<rocblas_int>(&argus.singular)->default_value(0),
            "Test with degenerate matrices? 0 = No, 1 = Yes\n"
            "                           This will produce matrices that are singular, non positive-definite, etc.\n"
            "                           ")

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
            "Validate GPU results with CPU? 0 = No, 1 = Yes.\n"
            "                           This will additionally print the relative error of the computations.\n"
            "                           ")

        // size options
        ("k",
         value<rocblas_int>()->default_value(128),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of Householder reflections in a transformation.\n"
            "                           ")

        ("m",
         value<rocblas_int>()->default_value(128),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of rows of a matrix.\n"
            "                           ")

        ("n",
         value<rocblas_int>()->default_value(128),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of columns of a matrix,\n"
            "                           or the order of a system or transformation.\n"
            "                           ")

        ("nrhs",
         value<rocblas_int>()->default_value(128),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of columns of a matrix on the right-hand side.\n"
            "                           ")

        // leading dimension options
        ("lda",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices A.\n"
            "                           ")

        ("ldb",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices B.\n"
            "                           ")

        ("ldc",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices C.\n"
            "                           ")

        ("ldt",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices T.\n"
            "                           ")

        ("ldu",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices U.\n"
            "                           ")

        ("ldv",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices V.\n"
            "                           ")

        ("ldw",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices W.\n"
            "                           ")

        ("ldx",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices X.\n"
            "                           ")

        ("ldy",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices Y.\n"
            "                           ")

        // stride options
        ("strideA",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors A.\n"
            "                           ")

        ("strideB",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors B.\n"
            "                           ")

        ("strideD",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors D.\n"
            "                           ")

        ("strideE",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors E.\n"
            "                           ")

        ("strideQ",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for vectors tau and ipiv.\n"
            "                           ")

        ("strideP",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for vectors tau and ipiv.\n"
            "                           ")

        ("strideS",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors S.\n"
            "                           ")

        ("strideU",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors U.\n"
            "                           ")

        ("strideV",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors V.\n"
            "                           ")

        // bdsqr options
        ("nc",
         value<rocblas_int>()->default_value(128),
            "The number of columns of matrix C.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        ("nu",
         value<rocblas_int>()->default_value(128),
            "The number of columns of matrix U.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        ("nv",
         value<rocblas_int>()->default_value(128),
            "The number of columns of matrix V.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        // laswp options
        ("k1",
         value<rocblas_int>()->default_value(1),
            "First index for row interchange.\n"
            "                           Only applicable to laswp.\n"
            "                           ")

        ("k2",
         value<rocblas_int>()->default_value(2),
            "Last index for row interchange.\n"
            "                           Only applicable to laswp.\n"
            "                           ")

        // gesvd options
        ("fast_alg",
         value<char>()->default_value('O'),
            "Enables out-of-place computations.\n"
            "                           Only applicable to gesvd.\n"
            "                           ")

        ("left_svect",
         value<char>()->default_value('N'),
            "Computation type for left singular vectors.\n"
            "                           Only applicable to gesvd.\n"
            "                           ")

        ("right_svect",
         value<char>()->default_value('N'),
            "Computation type for right singular vectors.\n"
            "                           Only applicable to gesvd.\n"
            "                           ")

        // other options
        ("direct",
         value<char>()->default_value('F'),
            "F = forward, B = backward.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("evect",
         value<char>()->default_value('N'),
            "Computation type for eigenvectors.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("incx",
         value<rocblas_int>()->default_value(1),
            "Increment between values in vector x.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("itype",
         value<char>()->default_value('1'),
            "Problem type for generalized eigenproblems.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("side",
         value<char>()->default_value('L'),
            "L = left, R = right.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("storev",
         value<char>()->default_value('C'),
            "C = column_wise, R = row_wise.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("trans",
         value<char>()->default_value('N'),
            "N = no transpose, T = transpose, C = conjugate transpose.\n"
            "                           Only applicable to certain routines.\n"
            "                           ")

        ("uplo",
         value<char>()->default_value('U'),
            "U = upper, L = lower.\n"
            "                           Only applicable to certain routines.\n"
            "                           ");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    // print help message
    if(vm.count("help"))
    {
        rocsolver_cout
            << "\nrocSOLVER benchmark client help.\n\n"
            << "Usage: ./rocsolver-bench <options>\n\n"
            << "In addition to some common general options, the following list of options "
               "corresponds to all the parameters\n"
            << "that might be needed to test a given rocSOLVER function. The parameters are named "
               "as in the API user guide.\n"
            << "The arrays are initialized internally by the program with random values.\n\n"
            << "Note: When a required parameter/option is not provided, it will take the default "
               "value as listed below.\n"
            << "If no default value is defined, the program will try to calculate a suitable value "
               "depending on the context\n"
            << "of the problem and the tested function; if this is not possible, the program will "
               "abort with error.\n\n"
            << "Example: ./rocsolver-bench -f getf2_batched -m 30 --lda 75 --batch_count 350\n"
            << "This will test getf2_batched with a set of 350 random 30x128 matrices. strideP "
               "will "
               "be set to be equal to 30.\n\n"
            << "Options:\n"
            << desc << std::endl;
        return 0;
    }

    argus.populate(vm);

    // set device ID
    if(!argus.perf)
    {
        rocblas_int device_count = query_device_property();
        if(device_count <= device_id)
            throw std::invalid_argument("Invalid Device ID");
    }
    set_device(device_id);

    // catch invalid arguments
    try
    {
        argus.validate_precision("precision");
        argus.validate_operation("trans");
        argus.validate_side("side");
        argus.validate_fill("uplo");
        argus.validate_direct("direct");
        argus.validate_storev("storev");
        argus.validate_svect("left_svect");
        argus.validate_svect("right_svect");
        argus.validate_workmode("fast_alg");
        argus.validate_evect("evect");
        argus.validate_itype("itype");
    }
    catch(const std::invalid_argument& e)
    {
        argus.clear();
        throw e;
    }

    // select and dispatch function test/benchmark
    rocsolver_dispatcher::invoke(function, precision, argus);

    return 0;
}

catch(const std::invalid_argument& exp)
{
    rocsolver_cerr << exp.what() << std::endl;
    return -1;
}
