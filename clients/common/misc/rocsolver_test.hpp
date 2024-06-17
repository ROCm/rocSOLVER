/* **************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <stdexcept>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <rocblas/rocblas.h>

// If USE_ROCBLAS_REALLOC_ON_DEMAND is false, automatic reallocation is disable and we will manually
// reallocate workspace
#define USE_ROCBLAS_REALLOC_ON_DEMAND true

#ifdef ROCSOLVER_CLIENTS_TEST
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)                                                       \
    {                                                                                                 \
        ASSERT_LE((max_error), (tol)*get_epsilon<T>());                                               \
        std::cout << "[          ] " << "Error / (K * n * ulp) <= "  << (                             \
                (tol > get_safemin<T>()) ? max_error/(tol * get_epsilon<T>()) : get_safemin<T>()      \
        ) << " [number K is test dependent]" << std::endl << std::flush;  \
    }                                                                                                 \

#else // ROCSOLVER_CLIENTS_BENCH
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)
#endif

typedef enum rocsolver_inform_type_
{
    inform_quick_return,
    inform_invalid_size,
    inform_invalid_args,
    inform_mem_query,
} rocsolver_inform_type;

inline void rocsolver_bench_inform(rocsolver_inform_type it, size_t arg = 0)
{
    switch(it)
    {
    case inform_quick_return: fmt::print("Quick return...\n"); break;
    case inform_invalid_size: fmt::print("Invalid size arguments...\n"); break;
    case inform_invalid_args: fmt::print("Invalid value in arguments...\n"); break;
    case inform_mem_query: fmt::print("{} bytes of device memory are required...\n", arg); break;
    }
    fmt::print("No performance data to collect.\n");
    fmt::print("No computations to verify.\n");
    std::fflush(stdout);
}

// recursive format function (base case)
inline void format_bench_table(std::string&) {}

// recursive format function
template <typename T, typename... Ts>
inline void format_bench_table(std::string& str, T arg, Ts... args)
{
    str += fmt::format("{:<15}", arg);
    if(sizeof...(Ts) > 0)
        str += ' ';
    format_bench_table(str, args...);
}

template <typename... Ts>
void rocsolver_bench_output(Ts... args)
{
    std::string table_row;
    format_bench_table(table_row, args...);
    std::puts(table_row.c_str());
    std::fflush(stdout);
}

inline void rocsolver_bench_header(const char* title)
{
    fmt::print("\n{:=<44}\n{}\n{:=<44}\n", "", title, "");
}

inline void rocsolver_bench_endl()
{
    std::putc('\n', stdout);
    std::fflush(stdout);
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return scalar;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return std::conj(scalar);
}

// A struct implicity convertable to and from char, used so we can customize Google Test
// output for LAPACK char arguments without affecting the default char output.
class printable_char
{
    char value;

public:
    printable_char(char c)
        : value(c)
    {
        if(c < 0x20 || c >= 0x7F)
            throw std::invalid_argument(fmt::format(
                "printable_char must be a printable ASCII character (received {:#x})", c));
    }

    operator char() const
    {
        return value;
    }
};

// gtest printers

inline std::ostream& operator<<(std::ostream& os, rocblas_status x)
{
    return os << rocblas_status_to_string(x);
}

inline std::ostream& operator<<(std::ostream& os, printable_char x)
{
    return os << char(x);
}

// location of the sparse data directory for the re-factorization tests
fs::path get_sparse_data_dir();

// Hash arrays following the spirit of boost::hash_combine
template<typename T>
std::size_t hash_combine(std::size_t seed, T value)
{
    using S = decltype(std::real(T{}));
    auto hasher = std::hash<S>();

    if constexpr (rocblas_is_complex<T>)
    {
        seed ^= hasher(std::real(value)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(std::imag(value)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    else
    {
        seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    return seed;
}

template<typename T>
std::size_t hash_combine(std::size_t seed, T const *array, std::size_t array_size)
{
    std::size_t hash = hash_combine(seed, array_size);
    for (std::size_t i = 0; i < array_size; ++i)
    {
        hash = hash_combine(hash, array[i]);
    }

    return hash;
}

template<typename T>
std::size_t hash_combine(std::size_t seed, const std::vector<T>& array)
{
    return hash_combine(seed, array.data(), array.size());
}

#include "common/misc/lapack_host_reference.hpp"

// Compute eigenvalues and eigenvectors of A with lapack_*syev
template<typename T, typename S>
bool eig(const rocblas_fill uplo, T const *A, const int n, T *U, S *D)
{
    if (A == nullptr || n < 1)
    {
        return false;
    }
    [[maybe_unused]] auto mptr = memcpy(U, A, n * n * sizeof(T));

    int info;
    int worksize = n * n;
    std::vector<T> work(worksize, T(0.));
    int worksize_real = n * n;
    std::vector<S> work_real(worksize_real, S(0.));
    cpu_syev_heev(rocblas_evect_original, uplo, n, U, n, D, work.data(), worksize, work_real.data(), worksize_real, &info);

    return (info == 0);
}

template<typename T, typename S>
bool eig(const rocblas_fill uplo, T const *A, const int n, std::vector<T> &U, std::vector<S> &D)
{
    if (A == nullptr || n < 1)
    {
        return false;
    }

    D.resize(n, S(0.));
    U.resize(n * n, T(0.));

    return eig(uplo, A, n, D.data(), U.data());
}

// Form matrix Y = X * D * X^*, where X^* is the adjoint of X
// X is nrowsX x dimD; D is dimD x dimD; Y is nrowsX x nrowsX
template<typename T, typename S>
bool XDXh(T const *X, const int nrowsX, S const *D, const int dimD, T *Y)
{
    if (X == nullptr || D == nullptr)
    {
        return false;
    }

    const int ldX = nrowsX;
    constexpr bool T_is_complex = rocblas_is_complex<T>;
    auto rocblas_operation_adjoint = T_is_complex ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    std::vector<T> W(dimD * dimD, T(0.)), Z(nrowsX * dimD, T(0.));
    const int ldZ = nrowsX;
    const int ldU = dimD;
    for (int i = 0; i < dimD; ++i)
    {
        W[i + i * dimD] = D[i];
    }

    cpu_gemm(rocblas_operation_none, rocblas_operation_none, nrowsX, dimD, dimD, T(1.), const_cast<T*>(X), nrowsX, W.data(), dimD, T(0.), Z.data(), dimD);
    cpu_gemm(rocblas_operation_none, rocblas_operation_adjoint, nrowsX, dimD, nrowsX, T(1.), Z.data(), nrowsX, const_cast<T*>(X), dimD, T(0.), Y, dimD);
    return true;
}

template<typename T, typename S>
bool XDXh(T const *X, const int nrows, S const *D, const int dimD, std::vector<T> &Y)
{
    Y.resize(dimD * dimD, T(0.));
    return XDXh(X, nrows, D, dimD, Y.data());
}

// Form matrix Y = X^* * X - I, where X^* is the adjoint of X
// X is nrows x ncols; Y is nrows x nrows
template<typename T>
bool XhXminusI(T const *X, const int nrows, const int ncols, std::vector<T> &Y)
{
    if (X == nullptr)
    {
        return false;
    }

    auto rocblas_operation_adjoint = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
    Y.resize(ncols * ncols, T(0.));
    cpu_gemm(rocblas_operation_adjoint, rocblas_operation_none, ncols, nrows, ncols, T(1.), X, ncols, X, nrows, T(0.), Y.data(), ncols);

    for (int i = 0; i < ncols; ++i)
    {
        Y[i + i * ncols] -= T(1.);
    }

    return true;
}

//
// Given inputs X (size nrowsX x dimD) and D (size dimD):
//
// Form matrix Y = U * diag(D) * U^*, where
// - U is the unitary matrix obtained from the QR decomposition of X,
// - U^* is the adjoint of U.
//
// Output Y is nrowsU x nrowsU
//
template<typename T, typename S>
bool UDUh(T const *X, const int nrowsX, S const *D, const int dimD, T *Y)
{
    const int ncolsX = dimD;
    const int ldX = nrowsX;
    const int m = std::max(nrowsX, ncolsX);

    int info;
    int worksize = 1;
    std::vector<T> work(worksize, T(0.)); // lapack workspace
    std::vector<T> tau(m); // scalar factors of geqrf reflectors
    auto rocblas_operation_adjoint = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    //
    // Create diagonal matrix W = diag(D)
    //
    const int dimW = dimD;
    std::vector<T> W(dimW * dimW, T(0.)), Z(dimW * dimW, T(0.));
    for (int i = 0; i < dimW; ++i)
    {
        W[i + i * dimW] = D[i];
    }

    //
    // Extract unitary matrix
    //
    std::vector<T> U(nrowsX * ncolsX, T(0.));
    { [[maybe_unused]] auto mptr = memcpy(U.data(), X, nrowsX * ncolsX * sizeof(T)); }

    // Pick something that is big enough for work size of geqrf
    worksize = m * m;
    work.resize(worksize, T(0.));
    const int nrowsU = nrowsX;
    const int ncolsU = ncolsX;
    const int ldU = ldX;
    cpu_geqrf<T>(nrowsU, ncolsU, U.data(), ldX, tau.data(), work.data(), worksize);

    /* // Infer work size of [or,un]mqr */
    /* worksize = -1; */
    /* cpu_ormqr_unmqr<T>(rocblas_side_right, rocblas_operation_adjoint, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info); */
    info = -1;

    if (info == 0)
    {
        // Use LAPACK's suggested work size
        worksize = std::real(work[0]);
    }
    else
    {
        // Pick something that is big enough for work size
        worksize = m * m;
    }
    work.resize(worksize, T(0.));

    //
    // Create matrix: Y = U * diag(D) * U^*
    //
    cpu_ormqr_unmqr<T>(rocblas_side_left, rocblas_operation_none, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info);
    cpu_ormqr_unmqr<T>(rocblas_side_right, rocblas_operation_adjoint, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info);
    { [[maybe_unused]] auto mptr = memcpy(Y, W.data(), W.size() * sizeof(T)); }

    return true;
}
