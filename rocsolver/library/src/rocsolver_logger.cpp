/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver_logger.hpp"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sys/time.h>

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

// initialize the static variable
rocsolver_logger* rocsolver_logger::_instance = nullptr;
std::mutex rocsolver_logger::_mutex;

/***************************************************************************
 * Open logging streams
 ***************************************************************************/

std::unique_ptr<rocsolver_ostream>
    rocsolver_logger::open_log_stream(const char* environment_variable_name)
{
    const char* logfile;
    if((logfile = std::getenv(environment_variable_name)) != nullptr
       || (logfile = std::getenv("ROCSOLVER_LOG_PATH")) != nullptr)
    {
        auto os = std::make_unique<rocsolver_ostream>(logfile);

        // print version info only once per file
        if(os != trace_os && os != bench_os && os != profile_os)
        {
            *os << "ROCSOLVER LOG FILE" << '\n';
            *os << "rocSOLVER Version: " << STRINGIFY(ROCSOLVER_VERSION_MAJOR) << '.'
                << STRINGIFY(ROCSOLVER_VERSION_MINOR) << '.' << STRINGIFY(ROCSOLVER_VERSION_PATCH)
                << '.' << STRINGIFY(ROCSOLVER_VERSION_TWEAK) << '\n';
            *os << "rocBLAS Version: " << STRINGIFY(ROCBLAS_VERSION_MAJOR) << '.'
                << STRINGIFY(ROCBLAS_VERSION_MINOR) << '.' << STRINGIFY(ROCBLAS_VERSION_PATCH)
                << '.' << STRINGIFY(ROCBLAS_VERSION_TWEAK) << '\n';
            *os << std::endl;
        }

        return os;
    }
    else
        return std::make_unique<rocsolver_ostream>(STDERR_FILENO);
}

/***************************************************************************
 * Call stack manipulation
 ***************************************************************************/

rocsolver_log_entry& rocsolver_logger::push_log_entry(rocblas_handle handle, std::string&& name)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    stack.push_back(rocsolver_log_entry());

    rocsolver_log_entry& result = stack.back();
    result.name = std::move(name);
    result.level = stack.size() - 1;
    result.start_time = get_time_us_no_sync();

    for(int i = 1; i < stack.size() - 1; i++)
        result.callers.push_back(stack[i].name);

    return result;
}

rocsolver_log_entry& rocsolver_logger::peek_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    rocsolver_log_entry& result = stack.back();
    return result;
}

rocsolver_log_entry rocsolver_logger::pop_log_entry(rocblas_handle handle)
{
    std::vector<rocsolver_log_entry>& stack = call_stack[handle];
    rocsolver_log_entry result = stack.back();
    stack.pop_back();

    if(stack.empty())
        call_stack.erase(handle);

    return result;
}

/***************************************************************************
 * Profile log printing
 ***************************************************************************/

void rocsolver_logger::write_profile(rocsolver_profile_map::iterator start,
                                     rocsolver_profile_map::iterator end)
{
    for(auto it = start; it != end; ++it)
    {
        rocsolver_profile_entry& entry = it->second;
        for(int i = 0; i < entry.level - 1; i++)
            *profile_os << "    ";

        *profile_os << it->first << ": Calls: " << entry.calls
                    << ", Total Time: " << (entry.time * 0.001) << " ms";

        if(entry.internal_calls)
        {
            double internal_time = 0;
            for(const auto& nested : *entry.internal_calls)
                internal_time += nested.second.time;

            *profile_os << " (in nested functions: " << (internal_time * 0.001) << " ms)" << '\n';

            write_profile(entry.internal_calls->begin(), entry.internal_calls->end());
        }
        else
            *profile_os << '\n';
    }
}

rocblas_status rocsolver_log_write_profile(void)
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        *logger->profile_os << "------- PROFILE -------" << '\n';
        logger->write_profile(logger->profile.begin(), logger->profile.end());
        *logger->profile_os << std::endl;
    }

    return rocblas_status_success;
}

rocblas_status rocsolver_log_flush_profile(void)
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // print and clear profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        *logger->profile_os << "------- PROFILE -------" << '\n';
        logger->write_profile(logger->profile.begin(), logger->profile.end());
        *logger->profile_os << std::endl;

        logger->profile.clear();
    }

    return rocblas_status_success;
}

/***************************************************************************
 * Logging set-up and tear-down
 ***************************************************************************/

extern "C" {

rocblas_status rocsolver_log_begin()
try
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is no logger, create one and:
    if(rocsolver_logger::_instance != nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance = new rocsolver_logger();

    // set layer_mode from environment variable ROCSOLVER_LAYER or to default
    if(const char* str_layer_mode = std::getenv("ROCSOLVER_LAYER"))
    {
        errno = 0;
        long value = strtol(str_layer_mode, 0, 0);
        if(errno || value < 0 || size_t(value) > size_t(UINT32_MAX))
            return rocblas_status_internal_error;
        else
            logger->layer_mode = static_cast<rocblas_layer_mode_flags>(value);
    }
    else
        logger->layer_mode = rocblas_layer_mode_none;

    // set max_levels from value of environment variable ROCSOLVER_LEVELS or to default
    if(const char* str_max_level = std::getenv("ROCSOLVER_LEVELS"))
    {
        errno = 0;
        long value = strtol(str_max_level, 0, 0);
        if(errno || value < 1 || size_t(value) > size_t(INT_MAX))
            return rocblas_status_internal_error;
        else
            logger->max_levels = static_cast<int>(value);
    }
    else
        logger->max_levels = 1;

    // create output streams (specified by env variables or default to stderr)
    logger->trace_os = logger->open_log_stream("ROCSOLVER_LOG_TRACE_PATH");
    logger->bench_os = logger->open_log_stream("ROCSOLVER_LOG_BENCH_PATH");
    logger->profile_os = logger->open_log_stream("ROCSOLVER_LOG_PROFILE_PATH");

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocsolver_log_end()
try
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // if there are pending log_exit calls:
    if(!rocsolver_logger::_instance->call_stack.empty())
        return rocblas_status_internal_error;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && !logger->profile.empty())
    {
        *logger->profile_os << "------- PROFILE -------" << '\n';
        logger->write_profile(logger->profile.begin(), logger->profile.end());
        *logger->profile_os << std::endl;
    }

    // delete the logger
    delete rocsolver_logger::_instance;
    rocsolver_logger::_instance = nullptr;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocsolver_log_set_layer_mode(const rocblas_layer_mode_flags layer_mode)
try
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // change to user specified mode.
    // output streams remain the same defined at logger creation
    logger->layer_mode = layer_mode;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocsolver_log_set_max_levels(const rocblas_int max_levels)
try
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;
    if(max_levels < 1)
        return rocblas_status_invalid_value;

    auto logger = rocsolver_logger::_instance;

    // change to user specified levels.
    // output streams remain the same defined at logger creation
    logger->max_levels = max_levels;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}

rocblas_status rocsolver_log_restore_defaults(void)
try
{
    const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

    // if there is an active logger:
    if(rocsolver_logger::_instance == nullptr)
        return rocblas_status_internal_error;

    auto logger = rocsolver_logger::_instance;

    // reset to no logging
    logger->max_levels = 1;
    logger->layer_mode = rocblas_layer_mode_none;

    return rocblas_status_success;
}
catch(...)
{
    return exception_to_rocblas_status();
}
}

/***************************************************************************
 * rocsolver_logger member functions
 ***************************************************************************/
template <>
char rocsolver_logger::get_precision<float>()
{
    return 's';
}
template <>
char rocsolver_logger::get_precision<double>()
{
    return 'd';
}
template <>
char rocsolver_logger::get_precision<rocblas_float_complex>()
{
    return 'c';
}
template <>
char rocsolver_logger::get_precision<rocblas_double_complex>()
{
    return 'z';
}

double rocsolver_logger::get_time_us()
{
    hipDeviceSynchronize();
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

double rocsolver_logger::get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}

double rocsolver_logger::get_time_us_no_sync()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_sec * 1'000'000llu + (tv.tv_nsec + 500llu) / 1000;
}
