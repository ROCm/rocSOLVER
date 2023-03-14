/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <rocblas/rocblas.h>

/* The rocsolver library maintains binary compatibility when built without rocsparse,
   but the functions that depend on rocsparse functionality will return
   rocblas_status_not_implemented.
*/

extern "C" {

rocblas_status rocsolver_rfinfo_create(rocsolver_rfinfo*, rocblas_handle)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_rfinfo_destroy(rocsolver_rfinfo)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_scsrrf_analysis(rocblas_handle,
                                         const rocblas_int,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         float*,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         float*,
                                         rocblas_int*,
                                         rocblas_int*,
                                         rocsolver_rfinfo)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_dcsrrf_analysis(rocblas_handle,
                                         const rocblas_int,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         double*,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         double*,
                                         rocblas_int*,
                                         rocblas_int*,
                                         rocsolver_rfinfo)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_scsrrf_refactlu(rocblas_handle,
                                         const rocblas_int,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         float*,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         float*,
                                         rocblas_int*,
                                         rocblas_int*,
                                         rocsolver_rfinfo)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_dcsrrf_refactlu(rocblas_handle,
                                         const rocblas_int,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         double*,
                                         const rocblas_int,
                                         rocblas_int*,
                                         rocblas_int*,
                                         double*,
                                         rocblas_int*,
                                         rocblas_int*,
                                         rocsolver_rfinfo)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_scsrrf_splitlu(rocblas_handle,
                                        const rocblas_int,
                                        const rocblas_int,
                                        rocblas_int*,
                                        rocblas_int*,
                                        float*,
                                        rocblas_int*,
                                        rocblas_int*,
                                        float*,
                                        rocblas_int*,
                                        rocblas_int*,
                                        float*)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_dcsrrf_splitlu(rocblas_handle,
                                        const rocblas_int,
                                        const rocblas_int,
                                        rocblas_int*,
                                        rocblas_int*,
                                        double*,
                                        rocblas_int*,
                                        rocblas_int*,
                                        double*,
                                        rocblas_int*,
                                        rocblas_int*,
                                        double*)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_scsrrf_sumlu(rocblas_handle,
                                      const rocblas_int,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      float*,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      float*,
                                      rocblas_int*,
                                      rocblas_int*,
                                      float*)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_dcsrrf_sumlu(rocblas_handle,
                                      const rocblas_int,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      double*,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      double*,
                                      rocblas_int*,
                                      rocblas_int*,
                                      double*)
{
    return rocblas_status_not_implemented;
}

} // extern C
