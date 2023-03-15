/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver/rocsolver.h"

/* The rocsolver library maintains binary compatibility when built without rocsparse,
   but the functions that depend on rocsparse functionality will return
   rocblas_status_not_implemented.
*/

extern "C" {

rocblas_status rocsolver_create_rfinfo(rocsolver_rfinfo*, rocblas_handle)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_destroy_rfinfo(rocsolver_rfinfo)
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

rocblas_status rocsolver_scsrrf_solve(rocblas_handle,
                                      const rocblas_int,
                                      const rocblas_int,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      float*,
                                      rocblas_int*,
                                      rocblas_int*,
                                      rocsolver_rfinfo,
                                      float*,
                                      const rocblas_int)
{
    return rocblas_status_not_implemented;
}

rocblas_status rocsolver_dcsrrf_solve(rocblas_handle,
                                      const rocblas_int,
                                      const rocblas_int,
                                      const rocblas_int,
                                      rocblas_int*,
                                      rocblas_int*,
                                      double*,
                                      rocblas_int*,
                                      rocblas_int*,
                                      rocsolver_rfinfo,
                                      double*,
                                      const rocblas_int)
{
    return rocblas_status_not_implemented;
}

} // extern C
