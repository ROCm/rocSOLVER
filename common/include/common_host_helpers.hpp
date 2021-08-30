/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <hip/hip_runtime_api.h>

#include "rocblascommon/utility.hpp"

/*
 * ===========================================================================
 *    common location for functions that are used by both the rocSOLVER
 *    library and rocSOLVER client code.
 * ===========================================================================
 */

/* =============================================================================================== */
/* Timing functions.                                                                               */

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and
 * return wall time */
double get_time_us();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and
 * return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall
 * time */
double get_time_us_no_sync();

/* =============================================================================================== */
/* Print functions.                                                                                */
inline void pairs_to_string(std::string& str, const char* sep)
{
    // do nothing
}
template <typename T1, typename T2, typename... Ts>
void pairs_to_string(std::string& str, const char* sep, T1 arg1, T2 arg2, Ts... args)
{
    str += fmt::format("{} {}", arg1, arg2);

    if(sizeof...(Ts) > 0)
    {
        str += sep;
        pairs_to_string(str, sep, args...);
    }
}

/** Set of helpers to print out data hosted in the CPU and/or the GPU **/
/***********************************************************************/

/*! \brief Print provided data into specified stream (real case)*/
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_to_stream(std::ostream& os,
                     const std::string name,
                     const rocblas_int m,
                     const rocblas_int n,
                     T* A,
                     const rocblas_int lda)
{
    std::string s;
    bool empty = name.empty();
    if(!empty)
        s += fmt::format("{}-by-{} matrix: {}\n", m, n, name);
    for(int i = 0; i < m; i++)
    {
        if(!empty)
            s += "    ";
        for(int j = 0; j < n; j++)
        {
            s += fmt::format("{}", A[j * lda + i]);
            if(j < n - 1)
                s += ", ";
        }
        s += '\n';
    }
    s += '\n';
    os << s;
    os.flush();
}

/*! \brief Print provided data into specified stream (complex cases)*/
template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
void print_to_stream(std::ostream& os,
                     const std::string name,
                     const rocblas_int m,
                     const rocblas_int n,
                     T* A,
                     const rocblas_int lda)
{
    std::string s;
    bool empty = name.empty();
    if(!empty)
        s += fmt::format("{}-by-{} matrix: {}\n", m, n, name);
    for(int i = 0; i < m; i++)
    {
        if(!empty)
            s += "    ";
        for(int j = 0; j < n; j++)
        {
            s += fmt::format("[{}+{}i]", A[j * lda + i].real(), A[j * lda + i].imag());
            if(j < n - 1)
                s += ", ";
        }
        s += '\n';
    }
    s += '\n';
    os << s;
    os.flush();
}

/*! \brief Print data from a normal or strided_batched array on the GPU to screen*/
template <typename T>
void print_device_matrix(std::ostream& os,
                         const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* A,
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0)
{
    std::vector<T> hA(lda * n);
    hipMemcpy(hA.data(), A + idx * stride, sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    print_to_stream<T>(os, name, m, n, hA.data(), lda);
}

/*! \brief Print data from a batched array on the GPU to screen*/
template <typename T>
void print_device_matrix(std::ostream& os,
                         const std::string name,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* const A[],
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0)
{
    std::vector<T> hA(lda * n);
    T* AA[1];
    hipMemcpy(AA, A + idx, sizeof(T*), hipMemcpyDeviceToHost);
    hipMemcpy(hA.data(), AA[0], sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    print_to_stream<T>(os, name, m, n, hA.data(), lda);
}

/*! \brief Print data from a normal or strided_batched array on the GPU to file*/
template <typename T>
void print_device_matrix(const std::string file,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* A,
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0)
{
    std::ofstream os(file);
    std::vector<T> hA(lda * n);
    hipMemcpy(hA.data(), A + idx * stride, sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    print_to_stream<T>(os, "", m, n, hA.data(), lda);
}

/*! \brief Print data from a batched array on the GPU to file*/
template <typename T>
void print_device_matrix(const std::string file,
                         const rocblas_int m,
                         const rocblas_int n,
                         T* const A[],
                         const rocblas_int lda,
                         const rocblas_stride stride = 1,
                         const rocblas_int idx = 0)
{
    std::ofstream os(file);
    std::vector<T> hA(lda * n);
    T* AA[1];
    hipMemcpy(AA, A + idx, sizeof(T*), hipMemcpyDeviceToHost);
    hipMemcpy(hA.data(), AA[0], sizeof(T) * lda * n, hipMemcpyDeviceToHost);

    print_to_stream<T>(os, "", m, n, hA.data(), lda);
}

/*! \brief Print data from a normal or strided_batched array on the CPU to screen*/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* A,
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0)
{
    print_to_stream<T>(os, name, m, n, A + idx * stride, lda);
}

/*! \brief Print data from a batched array on the CPU to screen*/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* const A[],
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0)
{
    print_to_stream<T>(os, name, m, n, A[idx], lda);
}

/*! \brief Print data from a normal or strided_batched array on the CPU to file*/
template <typename T>
void print_host_matrix(const std::string file,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* A,
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0)
{
    std::ofstream os(file);
    print_to_stream<T>(os, "", m, n, A + idx * stride, lda);
}

/*! \brief Print data from a batched array on the CPU to file*/
template <typename T>
void print_host_matrix(const std::string file,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* const A[],
                       const rocblas_int lda,
                       const rocblas_stride stride = 1,
                       const rocblas_int idx = 0)
{
    std::ofstream os(file);
    print_to_stream<T>(os, "", m, n, A[idx], lda);
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix */
/*******************************************************************/
template <typename T>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* CPU_result,
                       T* GPU_result,
                       const rocblas_int lda)
{
    std::string s;
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            s += fmt::format("matrix  col {}, row {}, CPU result={}, GPU result={}\n", i, j,
                             CPU_result[j + i * lda], GPU_result[j + i * lda]);
        }
    }
    s += '\n';
    os << s;
    os.flush();
}

template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* CPU_result,
                       T* GPU_result,
                       const rocblas_int lda,
                       double error_tolerance)
{
    std::string s;
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            T comp = (CPU_result[j + i * lda] - GPU_result[j + i * lda]) / CPU_result[j + i * lda];
            if(abs(comp) > error_tolerance)
                s += fmt::format("matrix  col {}, row {}, CPU result={}, GPU result={}\n", i, j,
                                 CPU_result[j + i * lda], GPU_result[j + i * lda]);
        }
    }
    s += '\n';
    os << s;
    os.flush();
}

template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
void print_host_matrix(std::ostream& os,
                       const std::string name,
                       const rocblas_int m,
                       const rocblas_int n,
                       T* CPU_result,
                       T* GPU_result,
                       const rocblas_int lda,
                       double error_tolerance)
{
    std::string s;
    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            T comp = (CPU_result[j + i * lda] - GPU_result[j + i * lda]) / CPU_result[j + i * lda];
            if(sqrt(comp.real() * comp.real() + comp.imag() * comp.imag()) > error_tolerance)
                s += fmt::format("matrix  col {}, row {}, CPU result={}, GPU result={}\n", i, j,
                                 CPU_result[j + i * lda], GPU_result[j + i * lda]);
        }
    }
    s += '\n';
    os << s;
    os.flush();
}
