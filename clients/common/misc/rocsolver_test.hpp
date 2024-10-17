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

#ifdef ROCSOLVER_CLIENTS_TEST_PRINT_EXTRA_MESSAGES
// Format output similarly as GTEST messages
#define ANSI_CODE_GTEST_GREEN "\033[0;32m"
#define ANSI_CODE_NORMAL_TERM "\033[0;0m"
// Macro ROCSOLVER_GTEST_MSG_PRINTER is also used to print hashes in tests
#define ROCSOLVER_GTEST_MSG_PRINTER \
    std::cout << ANSI_CODE_GTEST_GREEN << "[          ] " << ANSI_CODE_NORMAL_TERM
// Print computed errors for all tests, making sure that there are sufficient digits to uniquely
// represent all distinct `double` values.
#define ROCSOLVER_STRNGFY(s) #s
#define ROCSOLVER_PRINT_TEST_ERROR(T, max_error, tol) ROCSOLVER_PRINT_TEST_ERROR2(T, max_error, tol)
#define ROCSOLVER_PRINT_TEST_ERROR2(T, max_error, tol)                                  \
    do                                                                                  \
    {                                                                                   \
        const auto default_precision{std::cout.precision()};                            \
        constexpr auto max_precision{std::numeric_limits<double>::max_digits10 + 1};    \
        double tol_ = static_cast<double>(tol) * static_cast<double>(get_epsilon<T>()); \
        double max_error_ = static_cast<double>(max_error);                             \
        ROCSOLVER_GTEST_MSG_PRINTER                                                     \
            << "Computed error: " << ROCSOLVER_STRNGFY(max_error) << " / "              \
            << ROCSOLVER_STRNGFY(((tol)*get_epsilon<T>())) << " = "                     \
            << std::setprecision(max_precision)                                         \
            << ((max_error_ >= 0.) && (tol_ > 0.) ? max_error_ / tol_ : -1.)            \
            << std::setprecision(default_precision) << std::endl                        \
            << std::flush;                                                              \
    } while(0)
#else // #ifdef ROCSOLVER_CLIENTS_TEST_PRINT_EXTRA_MESSAGES
static std::stringstream rocsolver_discard_tests_extra_messages;
#define ROCSOLVER_GTEST_MSG_PRINTER rocsolver_discard_tests_extra_messages
#define ROCSOLVER_PRINT_TEST_ERROR(T, max_error, tol)
#endif // #ifdef ROCSOLVER_CLIENTS_TEST_PRINT_EXTRA_MESSAGES

#define ROCSOLVER_TEST_CHECK(T, max_error, tol)         \
    do                                                  \
    {                                                   \
        ASSERT_LE((max_error), (tol)*get_epsilon<T>()); \
        ROCSOLVER_PRINT_TEST_ERROR(T, max_error, tol);  \
    } while(0)

#else // ROCSOLVER_CLIENTS_BENCH
static std::stringstream rocsolver_discard_tests_extra_messages;
#define ROCSOLVER_GTEST_MSG_PRINTER rocsolver_discard_tests_extra_messages
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

/// Combines `seed` with the hash of `value`, following the spirit of
/// `boost::hash_combine`.
///
/// Extends `std::hash` to combine the hashes of multiple values (e.g.,
/// from an array).
///
/// Attention: hash_combine(0, T(0)) != 0
template <typename T>
std::size_t hash_combine(std::size_t seed, T value)
{
    using S = decltype(std::real(T{}));
    auto hasher = std::hash<S>();

    if constexpr(rocblas_is_complex<T>)
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

/// Hash contents of the given array.
///
/// If seed == 0 and array_size == 0, then hash_combine(seed, _, array_size) == 0
template <typename T>
std::size_t hash_combine(std::size_t seed, T const* array, std::size_t array_size)
{
    std::size_t hash = 0;
    if(array_size > 0)
    {
        hash = hash_combine(seed, array_size);
        for(std::size_t i = 0; i < array_size; ++i)
        {
            hash = hash_combine(hash, array[i]);
        }
    }

    return hash;
}

/// Hash contents of the given array.
///
/// If seed == 0 and array.size() == 0, then hash_combine(seed, array) == 0
template <typename T>
std::size_t hash_combine(std::size_t seed, const std::vector<T>& array)
{
    return hash_combine(seed, array.data(), array.size());
}
