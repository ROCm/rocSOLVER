/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef _ROCBLAS_HPP_
#define _ROCBLAS_HPP_

#include <rocblas.h>
#include "rocblas-exported-proto.hpp"
#include "helpers.h"
#include "ideal_sizes.hpp"
#include "common_device.hpp"

// scal
template <typename T, typename U, typename V>
rocblas_status rocblasCall_scal(rocblas_handle handle, 
                            rocblas_int    n, 
                            U              alpha,
                            rocblas_stride stridea,
                            V              x,
                            rocblas_int    offsetx,
                            rocblas_int    incx,
                            rocblas_stride stridex,
                            rocblas_int    batch_count)
{
    return rocblas_scal_template<ROCBLAS_SCAL_NB,T>(handle,n,alpha,stridea,x,offsetx,incx,stridex,batch_count);
} 

// dot
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_dot(rocblas_handle handle,
                           rocblas_int    n,
                           U              x,
                           rocblas_int    offsetx,
                           rocblas_int    incx,
                           rocblas_stride stridex,
                           U              y,
                           rocblas_int    offsety,
                           rocblas_int    incy,
                           rocblas_stride stridey,
                           rocblas_int    batch_count,
                           T*             results,
                           T*             workspace)
{
    return rocblas_dot_template<ROCBLAS_DOT_NB,CONJ,T>(handle,n,cast2constType<T>(x),offsetx,incx,stridex,
                                                       cast2constType<T>(y),offsety,incy,stridey,
                                                       batch_count,results,workspace);                         
}

// ger
template <bool CONJ, typename T, typename U, typename V>
rocblas_status rocblasCall_ger(rocblas_handle handle, 
                           rocblas_int    m, 
                           rocblas_int    n,
                           U              alpha, 
                           rocblas_stride stridea,
                           V              x,
                           rocblas_int    offsetx,
                           rocblas_int    incx,
                           rocblas_stride stridex,
                           V              y, 
                           rocblas_int    offsety,
                           rocblas_int    incy, 
                           rocblas_stride stridey,   
                           V              A, 
                           rocblas_int    offsetA,
                           rocblas_int    lda,
                           rocblas_stride strideA,
                           rocblas_int    batch_count,
                           T**            work)
{
    return rocblas_ger_template<CONJ,T>(handle,m,n,alpha,stridea,cast2constType<T>(x),offsetx,incx,stridex,
                                        cast2constType<T>(y),offsety,incy,stridey,A,offsetA,lda,strideA,batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle, 
                           rocblas_int    m, 
                           rocblas_int    n,
                           U              alpha, 
                           rocblas_stride stridea,
                           T *const       x[],
                           rocblas_int    offsetx,
                           rocblas_int    incx,
                           rocblas_stride stridex,
                           T*             y, 
                           rocblas_int    offsety,
                           rocblas_int    incy, 
                           rocblas_stride stridey,   
                           T *const       A[], 
                           rocblas_int    offsetA,
                           rocblas_int    lda,
                           rocblas_stride strideA,
                           rocblas_int    batch_count,
                           T**            work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,y,stridey,batch_count);
 
    return rocblas_ger_template<CONJ,T>(handle,m,n,alpha,stridea,cast2constType<T>(x),offsetx,incx,stridex,
                                          cast2constType<T>(work),offsety,incy,stridey,A,offsetA,lda,strideA,batch_count);
}

// ger overload
template <bool CONJ, typename T, typename U>
rocblas_status rocblasCall_ger(rocblas_handle handle, 
                           rocblas_int    m, 
                           rocblas_int    n,
                           U              alpha, 
                           rocblas_stride stridea,
                           T*             x,
                           rocblas_int    offsetx,
                           rocblas_int    incx,
                           rocblas_stride stridex,
                           T *const       y[], 
                           rocblas_int    offsety,
                           rocblas_int    incy, 
                           rocblas_stride stridey,   
                           T *const       A[], 
                           rocblas_int    offsetA,
                           rocblas_int    lda,
                           rocblas_stride strideA,
                           rocblas_int    batch_count,
                           T**            work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,x,stridex,batch_count);
 
    return rocblas_ger_template<CONJ,T>(handle,m,n,alpha,stridea,cast2constType<T>(work),offsetx,incx,stridex,
                                          cast2constType<T>(y),offsety,incy,stridey,A,offsetA,lda,strideA,batch_count);
}

// gemv
template<typename T, typename U, typename V>
rocblas_status rocblasCall_gemv(rocblas_handle    handle,
                            rocblas_operation transA,
                            rocblas_int       m,
                            rocblas_int       n,
                            U                 alpha,
                            rocblas_stride    stride_alpha,
                            V                 A,
                            rocblas_int       offseta,
                            rocblas_int       lda,
                            rocblas_stride    strideA,
                            V                 x,
                            rocblas_int       offsetx,
                            rocblas_int       incx,
                            rocblas_stride    stridex,
                            U                 beta,
                            rocblas_stride    stride_beta,
                            V                 y,
                            rocblas_int       offsety,
                            rocblas_int       incy,
                            rocblas_stride    stridey,
                            rocblas_int       batch_count,
                            T**               work)
{
    return rocblas_gemv_template<T>(handle,transA,m,n,alpha,stride_alpha,
                                    cast2constType<T>(A),offseta,lda,strideA,
                                    cast2constType<T>(x),offsetx,incx,stridex,
                                    beta,stride_beta,
                                    y,offsety,incy,stridey,batch_count);
}

// gemv overload
template<typename T, typename U>
rocblas_status rocblasCall_gemv(rocblas_handle    handle,
                            rocblas_operation transA,
                            rocblas_int       m,
                            rocblas_int       n,
                            U                 alpha,
                            rocblas_stride    stride_alpha,
                            T *const          A[],
                            rocblas_int       offseta,
                            rocblas_int       lda,
                            rocblas_stride    strideA,
                            T *const          x[],
                            rocblas_int       offsetx,
                            rocblas_int       incx,
                            rocblas_stride    stridex,
                            U                 beta,
                            rocblas_stride    stride_beta,
                            T*                y,
                            rocblas_int       offsety,
                            rocblas_int       incy,
                            rocblas_stride    stridey,
                            rocblas_int       batch_count,
                            T**               work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,y,stridey,batch_count);
 
    return rocblas_gemv_template<T>(handle,transA,m,n,alpha,stride_alpha,
                                      cast2constType<T>(A),offseta,lda,strideA,
                                      cast2constType<T>(x),offsetx,incx,stridex,beta,stride_beta,
                                      cast2constPointer<T>(work),offsety,incy,stridey,batch_count);
}

// trmv
template<typename T, typename U>
rocblas_status rocblasCall_trmv(rocblas_handle    handle,
                            rocblas_fill      uplo,
                            rocblas_operation transa,
                            rocblas_diagonal  diag,
                            rocblas_int       m,
                            U                 a,
                            rocblas_int       offseta,
                            rocblas_int       lda,
                            rocblas_stride    stridea,
                            U                 x,
                            rocblas_int       offsetx,
                            rocblas_int       incx,
                            rocblas_stride    stridex,
                            T*                w,
                            rocblas_stride    stridew,
                            rocblas_int       batch_count)
{
    return rocblas_trmv_template<ROCBLAS_TRMV_NB>(handle,uplo,transa,diag,m,cast2constType<T>(a),offseta,lda,stridea,
                                                  x,offsetx,incx,stridex,w,stridew,batch_count);
}

// gemm
template <bool BATCHED, bool STRIDED, typename T, typename U, typename V>
rocblas_status rocblasCall_gemm(rocblas_handle    handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            rocblas_int       m,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            V                 A,
                            rocblas_int       offset_a,
                            rocblas_int       ld_a,
                            rocblas_stride    stride_a,
                            V                 B,
                            rocblas_int       offset_b,
                            rocblas_int       ld_b,
                            rocblas_stride    stride_b,
                            U                 beta,
                            V                 C,
                            rocblas_int       offset_c,
                            rocblas_int       ld_c,
                            rocblas_stride    stride_c,
                            rocblas_int       batch_count,
                            T**               work)
{
    return rocblas_gemm_template<BATCHED,STRIDED,T>(handle,trans_a,trans_b,m,n,k,alpha,
                                                    cast2constType<T>(A),offset_a,ld_a,stride_a,
                                                    cast2constType<T>(B),offset_b,ld_b,stride_b,beta,
                                                    C,offset_c,ld_c,stride_c,batch_count);
}

//gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle    handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            rocblas_int       m,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            T*                A,
                            rocblas_int       offset_a,
                            rocblas_int       ld_a,
                            rocblas_stride    stride_a,
                            T *const          B[],
                            rocblas_int       offset_b,
                            rocblas_int       ld_b,
                            rocblas_stride    stride_b,
                            U                 beta,
                            T *const          C[],
                            rocblas_int       offset_c,
                            rocblas_int       ld_c,
                            rocblas_stride    stride_c,
                            rocblas_int       batch_count,
                            T**               work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,A,stride_a,batch_count);
 
    return rocblas_gemm_template<BATCHED,STRIDED,T>(handle,trans_a,trans_b,m,n,k,alpha,
                                                      cast2constType<T>(work),offset_a,ld_a,stride_a,
                                                      cast2constType<T>(B),offset_b,ld_b,stride_b,beta,
                                                      C,offset_c,ld_c,stride_c,batch_count);
}

//gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle    handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            rocblas_int       m,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            T *const          A[],
                            rocblas_int       offset_a,
                            rocblas_int       ld_a,
                            rocblas_stride    stride_a,
                            T*                B,
                            rocblas_int       offset_b,
                            rocblas_int       ld_b,
                            rocblas_stride    stride_b,
                            U                 beta,
                            T *const          C[],
                            rocblas_int       offset_c,
                            rocblas_int       ld_c,
                            rocblas_stride    stride_c,
                            rocblas_int       batch_count,
                            T**               work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,B,stride_b,batch_count);
 
    return rocblas_gemm_template<BATCHED,STRIDED,T>(handle,trans_a,trans_b,m,n,k,alpha,
                                                      cast2constType<T>(A),offset_a,ld_a,stride_a,
                                                      cast2constType<T>(work),offset_b,ld_b,stride_b,beta,
                                                      C,offset_c,ld_c,stride_c,batch_count);
}

//gemm overload
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocblasCall_gemm(rocblas_handle    handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            rocblas_int       m,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            T *const          A[],
                            rocblas_int       offset_a,
                            rocblas_int       ld_a,
                            rocblas_stride    stride_a,
                            T *const          B[],
                            rocblas_int       offset_b,
                            rocblas_int       ld_b,
                            rocblas_stride    stride_b,
                            U                 beta,
                            T*                C,
                            rocblas_int       offset_c,
                            rocblas_int       ld_c,
                            rocblas_stride    stride_c,
                            rocblas_int       batch_count,
                            T**               work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocks =  (batch_count - 1)/256 + 1;
    hipLaunchKernelGGL(get_array,dim3(blocks),dim3(256),0,stream,work,C,stride_c,batch_count);
 
    return rocblas_gemm_template<BATCHED,STRIDED,T>(handle,trans_a,trans_b,m,n,k,alpha,
                                                      cast2constType<T>(A),offset_a,ld_a,stride_a,
                                                      cast2constType<T>(B),offset_b,ld_b,stride_b,beta,
                                                      work,offset_c,ld_c,stride_c,batch_count);
}

// syrk
template <typename T, typename U, typename V>
rocblas_status rocblasCall_syrk(rocblas_handle    handle,
                            rocblas_fill      uplo,
                            rocblas_operation transA,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            V                 A,
                            rocblas_int       offsetA,
                            rocblas_int       lda,
                            rocblas_stride    strideA,
                            U                 beta,
                            V                 C,
                            rocblas_int       offsetC,
                            rocblas_int       ldc,
                            rocblas_stride    strideC,
                            rocblas_int       batch_count)
{
    return rocblas_syrk_template(handle,uplo,transA,n,k,cast2constType<T>(alpha),cast2constType<T>(A),offsetA,lda,strideA,
                                 cast2constType<T>(beta),C,offsetC,ldc,strideC,batch_count);
}

// herk
template <typename S, typename T, typename U, typename V, std::enable_if_t<!is_complex<T>, int> = 0>
rocblas_status rocblasCall_herk(rocblas_handle    handle,
                            rocblas_fill      uplo,
                            rocblas_operation transA,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            V                 A,
                            rocblas_int       offsetA,
                            rocblas_int       lda,
                            rocblas_stride    strideA,
                            U                 beta,
                            V                 C,
                            rocblas_int       offsetC,
                            rocblas_int       ldc,
                            rocblas_stride    strideC,
                            rocblas_int       batch_count)
{
    return rocblas_syrk_template(handle,uplo,transA,n,k,cast2constType<S>(alpha),cast2constType<T>(A),offsetA,lda,strideA,
                                 cast2constType<S>(beta),C,offsetC,ldc,strideC,batch_count);
}

template <typename S, typename T, typename U, typename V, std::enable_if_t<is_complex<T>, int> = 0>
rocblas_status rocblasCall_herk(rocblas_handle    handle,
                            rocblas_fill      uplo,
                            rocblas_operation transA,
                            rocblas_int       n,
                            rocblas_int       k,
                            U                 alpha,
                            V                 A,
                            rocblas_int       offsetA,
                            rocblas_int       lda,
                            rocblas_stride    strideA,
                            U                 beta,
                            V                 C,
                            rocblas_int       offsetC,
                            rocblas_int       ldc,
                            rocblas_stride    strideC,
                            rocblas_int       batch_count)
{
    return rocblas_herk_template(handle,uplo,transA,n,k,cast2constType<S>(alpha),cast2constType<T>(A),offsetA,lda,strideA,
                                 cast2constType<S>(beta),C,offsetC,ldc,strideC,batch_count);
}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// THESE SHOULD BE SUBTITUTED BY THEIR CORRESPONDING 
// ROCBLAS TEMPLATE FUNCTIONS ONCE THEY ARE EXPORTED
// ROCBLAS.CPP CAN BE ELIMINATED THEN

 
// nrm2
template <typename T1, typename T2>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n, const T1 *x,
                            rocblas_int incx, T2 *result);
/*template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const float* x, const rocblas_int incx, float* result) {
  return rocblas_snrm2(handle, n, x, incx, result);
}
template <>
rocblas_status rocblas_nrm2(rocblas_handle handle, rocblas_int n,
                            const double* x, const rocblas_int incx, double* result) {
  return rocblas_dnrm2(handle, n, x, incx, result);
}*/

// iamax
template <typename T>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n, const T *x,
                             rocblas_int incx, rocblas_int *result);
/*template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const float *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_isamax(handle, n, x, incx, result);
}
template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const double *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_idamax(handle, n, x, incx, result);
}
template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const rocblas_float_complex *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_icamax(handle, n, x, incx, result);
}
template <>
rocblas_status rocblas_iamax(rocblas_handle handle, rocblas_int n,
                             const rocblas_double_complex *x, rocblas_int incx,
                             rocblas_int *result) {
  return rocblas_izamax(handle, n, x, incx, result);
}*/

// trsm
template <typename T>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const T *alpha, T *A, rocblas_int lda, T *B,
                            rocblas_int ldb);
/*template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const float *alpha, float *A, rocblas_int lda,
                            float *B, rocblas_int ldb) {
  return rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,ldb);
}
template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const double *alpha, double *A, rocblas_int lda,
                            double *B, rocblas_int ldb) {
  return rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B,ldb);
}
template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const rocblas_float_complex *alpha, rocblas_float_complex *A, rocblas_int lda,
                            rocblas_float_complex *B, rocblas_int ldb) {
    return rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}
template <>
rocblas_status rocblas_trsm(rocblas_handle handle, rocblas_side side,
                            rocblas_fill uplo, rocblas_operation transA,
                            rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            const rocblas_double_complex *alpha, rocblas_double_complex *A, rocblas_int lda,
                            rocblas_double_complex *B, rocblas_int ldb) {
    return rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}*/

// trmm
template <typename T>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                            rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            T *alpha, T *A, rocblas_int lda, T* B, rocblas_int ldb);
/*template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                            rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            float *alpha, float *A, rocblas_int lda, float* B, rocblas_int ldb)
{
    return rocblas_strmm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}
template <>
rocblas_status rocblas_trmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
                            rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int n,
                            double *alpha, double *A, rocblas_int lda, double* B, rocblas_int ldb)
{
    return rocblas_dtrmm(handle,side,uplo,trans,diag,m,n,alpha,A,lda,B,ldb);
}*/



#endif // _ROCBLAS_HPP_
