/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include <fstream>
#include <vector>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <gtest/gtest.h>
#include <rocblas.h>
#include <rocsolver.h>

#include "client_environment_helpers.hpp"
#include "clientcommon.hpp"

using ::testing::Matcher;
using ::testing::MatchesRegex;

#ifdef __cpp_lib_experimental_filesystem
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

class checkin_misc_LOGGING : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipMalloc(&dA, sizeof(double) * stA * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dP, sizeof(rocblas_int) * stP * bc), hipSuccess);
        ASSERT_EQ(hipMalloc(&dinfo, sizeof(rocblas_int) * bc), hipSuccess);
    }

    void TearDown() override
    {
        ASSERT_EQ(hipFree(dA), hipSuccess);
        ASSERT_EQ(hipFree(dP), hipSuccess);
        ASSERT_EQ(hipFree(dinfo), hipSuccess);
    }

    unsigned int nondeterministic_value()
    {
        return rd();
    }

    std::random_device rd;

    double* dA;
    rocblas_int *dP, *dinfo;

    const rocblas_int m = 25;
    const rocblas_int n = 25;
    const rocblas_int lda = m;
    const rocblas_stride stA = lda * n;
    const rocblas_stride stP = n;
    const rocblas_int bc = 3;
};

static void verify_file(const fs::path& filepath, const std::vector<std::string>& expected_lines)
{
    std::ifstream logfile(filepath);
    ASSERT_TRUE(logfile.good()) << "which implies a failure to open " << filepath;
    std::string line;
    size_t line_number = 1;
    for(; std::getline(logfile, line); ++line_number)
    {
        ASSERT_LE(line_number, expected_lines.size())
            << "extra line containing '" << line << "' in " << filepath;
        const std::string& expected_pattern = expected_lines[line_number - 1];
        EXPECT_TRUE(Matcher<std::string>(MatchesRegex(expected_pattern)).Matches(line))
            << "Mismatch at line " << line_number << ":\n     Actual line: " << line
            << "\nExpected pattern: " << expected_pattern;
    }
    ASSERT_EQ(line_number - 1, expected_lines.size()) << "missing lines in " << filepath;
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_trace)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path trace_filepath = temp_dir / fmt::format("trace.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", trace_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*ENTER rocsolver_dgetrf_strided_batched trace tree.*",
        ".*getrf.*m: 25, n: 25, shiftA: 0, lda: 25, shiftP: 0, bc: 3.*",
        ".*EXIT rocsolver_dgetrf_strided_batched trace tree.*",
        "\\s*",
    };

    verify_file(trace_filepath, expected_lines);

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(trace_filepath));
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_path)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path trace_filepath
        = temp_dir / fmt::format("rocsolver_log_path.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PATH", trace_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    EXPECT_TRUE(fs::exists(trace_filepath));

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(trace_filepath));
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_bench)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path bench_filepath = temp_dir / fmt::format("bench.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_BENCH_PATH", bench_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_bench), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*rocsolver-bench -f getrf_strided_batched -r d -m 25 -n 25 --lda 25 --strideA 625 "
        "--strideP 25 --batch_count 3",
    };

    verify_file(bench_filepath, expected_lines);

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(bench_filepath));
}

TEST_F(checkin_misc_LOGGING, rocblas_layer_mode_log_profile)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path profile_filepath = temp_dir / fmt::format("profile.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH", profile_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };

    verify_file(profile_filepath, expected_lines);

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(profile_filepath));
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_write_profile)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path profile_filepath = temp_dir / fmt::format("profile.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH", profile_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_write_profile(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_write_profile(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };

    verify_file(profile_filepath, expected_lines);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
    if(!HasFailure())
        EXPECT_TRUE(fs::remove(profile_filepath));
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_flush_profile)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path profile_filepath = temp_dir / fmt::format("profile.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_PROFILE_PATH", profile_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(1), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_flush_profile(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 1, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
        ".*PROFILE.*",
        ".*getrf.*Calls: 2, Total Time: .+ .+ .in nested functions: .+ .+.",
        "\\s*",
    };

    verify_file(profile_filepath, expected_lines);

    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
    if(!HasFailure())
        EXPECT_TRUE(fs::remove(profile_filepath));
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_restore_defaults_resets_layer_mode)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path trace_filepath = temp_dir / fmt::format("trace.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", trace_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_restore_defaults(), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);
    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
    };

    verify_file(trace_filepath, expected_lines);

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(trace_filepath));
}

TEST_F(checkin_misc_LOGGING, rocsolver_log_restore_defaults_resets_max_levels)
{
    rocblas_handle handle;
    ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);

    fs::path temp_dir = fs::temp_directory_path();
    fs::path trace_filepath = temp_dir / fmt::format("trace.{}.log", nondeterministic_value());
    scoped_envvar logpath_variable("ROCSOLVER_LOG_TRACE_PATH", trace_filepath.c_str());

    ASSERT_EQ(rocsolver_log_begin(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_max_levels(2), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_restore_defaults(), rocblas_status_success);
    ASSERT_EQ(rocsolver_log_set_layer_mode(rocblas_layer_mode_log_trace), rocblas_status_success);
    ASSERT_EQ(rocsolver_dgetrf_strided_batched(handle, m, n, dA, lda, stA, dP, stP, dinfo, bc),
              rocblas_status_success);
    ASSERT_EQ(rocsolver_log_end(), rocblas_status_success);

    std::vector<std::string> expected_lines = {
        "ROCSOLVER LOG FILE",
        "rocSOLVER Version: .*",
        "rocBLAS Version: .*",
        ".*ENTER rocsolver_dgetrf_strided_batched trace tree.*",
        ".*getrf.*m: 25, n: 25, shiftA: 0, lda: 25, shiftP: 0, bc: 3.*",
        ".*EXIT rocsolver_dgetrf_strided_batched trace tree.*",
        "\\s*",
    };

    verify_file(trace_filepath, expected_lines);

    if(!HasFailure())
        EXPECT_TRUE(fs::remove(trace_filepath));
}
