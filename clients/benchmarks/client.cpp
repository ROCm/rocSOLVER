/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <boost/program_options.hpp>
#include <iostream>
#include <stdio.h>

#include "testing_getf2_getrf.hpp"
#include "testing_getrs.hpp"
#include "testing_potf2.hpp"
#include "utility.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  Arguments argus;
  
  //disable unit_check in client benchmark, it is only
  // used in gtest unit test
  argus.unit_check = 0; 

  // enable timing check,otherwise no performance data collected
  argus.timing = 1;

  std::string function;
  char precision;

  rocblas_int device_id;
  vector<rocblas_int> range = {-1, -1, -1};

  po::options_description desc("rocblas client command line options");
  desc.add_options()("help,h", "produces this help message")
      // clang-format off
        ("range",
         po::value<vector<rocblas_int>>(&range)->multitoken(),
         "Range matrix size testing: BLAS-3 benchmarking only. Accept three positive integers. "
         "Usage: "
         "--range start end step"
         ". e.g "
         "--range 100 1000 200"
         ". Diabled if not specified. If enabled, user specified m,n,k will be nullified")
        
        ("sizem,m",
         po::value<rocblas_int>(&argus.M)->default_value(1024),
         "Specific matrix size testing: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows.")
        
        ("sizen,n",
         po::value<rocblas_int>(&argus.N)->default_value(1024),
         "Specific matrix/vector size testing: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of columns")

        ("sizek,k",
         po::value<rocblas_int>(&argus.K)->default_value(1024),
         "Specific matrix size testing:sizek is only applicable to BLAS-3: the number of columns in "
         "A & C  and rows in B.")

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

        ("bsa",
         po::value<rocblas_int>(&argus.bsa)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsb",
         po::value<rocblas_int>(&argus.bsb)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("bsc",
         po::value<rocblas_int>(&argus.bsc)->default_value(1024*1024),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

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
        
        ("side",
         po::value<char>(&argus.side_option)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")
        
        ("uplo",
         po::value<char>(&argus.uplo_option)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm
                                                                     // xtrmm
        ("diag",
         po::value<char>(&argus.diag_option)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm
                                                                                          // xtrmm
        ("batch",
         po::value<rocblas_int>(&argus.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched routines") // xtrsm xtrmm xgemm

        ("verify,v",
         po::value<rocblas_int>(&argus.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         po::value<rocblas_int>(&argus.iters)->default_value(10),
         "Iterations to run inside timing loop")
        
        ("device",
         po::value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (precision != 'h' && precision != 's' && precision != 'd' &&
      precision != 'c' && precision != 'z') {
    std::cerr << "Invalid value for --precision" << std::endl;
    return -1;
  }

  // Device Query
  rocblas_int device_count = query_device_property();

  if (device_count <= device_id) {
    printf("Error: invalid device ID. There may not be such device ID. Will "
           "exit \n");
    return -1;
  } else {
    set_device(device_id);
  }
  /* ============================================================================================
   */
  if (argus.M < 0 || argus.N < 0 || argus.K < 0) {
    printf("Invalide matrix dimension\n");
  }

  argus.start = range[0];
  argus.step = range[1];
  argus.end = range[2];

  if (function == "potf2") {
    if (precision == 's')
      testing_potf2<float>(argus);
    else if (precision == 'd')
      testing_potf2<double>(argus);
  } 
  else if (function == "getf2") {
    if (precision == 's')
      testing_getf2_getrf<float>(argus,0);
    else if (precision == 'd')
      testing_getf2_getrf<double>(argus,0);
  } 
  else if (function == "getrf") {
    if (precision == 's')
      testing_getf2_getrf<float>(argus,1);
    else if (precision == 'd')
      testing_getf2_getrf<double>(argus,1);
  } 
  else if (function == "getrs") {
    if (precision == 's')
      testing_getrs<float>(argus);
    else if (precision == 'd')
      testing_getrs<double>(argus);
  } 
  else {
    printf("Invalid value for --function \n");
    return -1;
  }

  return 0;
}
