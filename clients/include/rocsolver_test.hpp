/* ************************************************************************
 * Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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

// If USE_ROCBLAS_REALLOC_ON_DEMAND is false, automatic reallocation is disable and we will manually
// reallocate workspace
#define USE_ROCBLAS_REALLOC_ON_DEMAND true

#ifdef ROCSOLVER_CLIENTS_TEST
#define ROCSOLVER_TEST_CHECK(T, max_error, tol) ASSERT_LE((max_error), (tol)*get_epsilon<T>())
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

inline fs::path get_sparse_data_dir()
{
    // first check an environment variable
    if(const char* datadir = std::getenv("ROCSOLVER_TEST_DATA"))
        return fs::path{datadir};

    fs::path p = fs::current_path();
    fs::path p_parent = p.parent_path();
    fs::path installed = p.root_directory() / "opt" / "rocm" / "share" / "rocsolver" / "test";
    fs::path exe_relative = fs::path(rocsolver_exepath()) / ".." / "share" / "rocsolver" / "test";

    // check relative to the current directory and relative to each parent
    while(p != p_parent)
    {
        fs::path candidate = p / "clients" / "sparsedata";
        if(fs::exists(candidate))
            return candidate;
        p = p_parent;
        p_parent = p.parent_path();
    }

    // check relative to the running executable
    if(fs::exists(exe_relative))
        return exe_relative;

    // check relative to default install path
    if(fs::exists(installed))
        return installed;

    fmt::print(
        stderr, "Warning: default sparse data directories ({}, {}) not found, defaulting to current working directory.\n",
        exe_relative, installed);

    return fs::current_path();
}
