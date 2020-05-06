/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */


#ifndef S_TEST_H_
#define S_TEST_H_


#define ROCSOLVER_BENCH_INFORM(case)                                    \
    do                                                                  \
    {                                                                   \
        if (case)                                                       \
            rocblas_cout << "Invalid size arguments..." << std::endl;   \
        else                                                            \
            rocblas_cout << "Quick return..." << std::endl;             \
        rocblas_cout << "No performance data to collect." << std::endl; \
        rocblas_cout << "No computations to verify." << std::endl;      \
    } while(0)



#endif
