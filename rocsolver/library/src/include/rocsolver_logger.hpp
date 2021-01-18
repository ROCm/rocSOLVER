/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "common_host_helpers.hpp"
#include "common_ostream_helpers.hpp"
#include "rocsolver.h"
#include <unordered_map>

/***************************************************************************
 * rocSOLVER logging macros
 ***************************************************************************/

#define ROCSOLVER_ENTER_TOP(name, ...)                                              \
    do                                                                              \
    {                                                                               \
        if(logger != nullptr && logger->is_logging_enabled())                       \
            logger->log_enter_top_level<T>(handle, "rocsolver", name, __VA_ARGS__); \
    } while(0)
#define ROCSOLVER_ENTER(name, ...)                                        \
    do                                                                    \
    {                                                                     \
        if(logger != nullptr && logger->is_logging_enabled())             \
            logger->log_enter<T>(handle, "rocsolver", name, __VA_ARGS__); \
    } while(0)
#define ROCBLAS_ENTER(name, ...)                                        \
    do                                                                  \
    {                                                                   \
        if(logger != nullptr && logger->is_logging_enabled())           \
            logger->log_enter<T>(handle, "rocblas", name, __VA_ARGS__); \
    } while(0)

#define ROCSOLVER_RETURN_TOP(name, ...)                               \
    do                                                                \
    {                                                                 \
        rocblas_status _status = __VA_ARGS__;                         \
        if(logger != nullptr && logger->is_logging_enabled())         \
            logger->log_exit_top_level<T>(handle, "rocsolver", name); \
        return _status;                                               \
    } while(0)
#define ROCSOLVER_RETURN(name, ...)                           \
    do                                                        \
    {                                                         \
        rocblas_status _status = __VA_ARGS__;                 \
        if(logger != nullptr && logger->is_logging_enabled()) \
            logger->log_exit<T>(handle, "rocsolver", name);   \
        return _status;                                       \
    } while(0)
#define ROCBLAS_RETURN(name, ...)                             \
    do                                                        \
    {                                                         \
        rocblas_status _status = __VA_ARGS__;                 \
        if(logger != nullptr && logger->is_logging_enabled()) \
            logger->log_exit<T>(handle, "rocblas", name);     \
        return _status;                                       \
    } while(0)

/***************************************************************************
 * The rocsolver_log_entry struct records function data for profile logging
 * purposes.
 ***************************************************************************/
struct rocsolver_log_entry
{
    std::string name;
    int calls;
    double time;
    double internal_time;

    rocsolver_log_entry()
        : calls(0)
        , time(0)
        , internal_time(0)
    {
    }
};

/***************************************************************************
 * The rocsolver_logger class provides functions to be called upon entering
 * or exiting a function that will output multi-level logging information.
 ***************************************************************************/
class rocsolver_logger
{
private:
    // profile logging data keyed by function name
    std::unordered_map<std::string, rocsolver_log_entry> profile;
    // function call stack keyed by handle
    std::unordered_map<rocblas_handle, std::vector<rocsolver_log_entry>> call_stack;
    // the maximum depth at which nested function calls will appear in the log
    int max_levels;
    // layer mode enum describing which logging facilities are enabled
    rocblas_layer_mode layer_mode;

    // convert type T into char s, d, c, or z
    template <typename T>
    char get_precision();

    // combines a function prefix and name into an std::string
    template <typename T>
    inline std::string get_func_name(const char* func_prefix, const char* func_name)
    {
        return std::string(func_prefix) + '_' + get_precision<T>() + func_name;
    }
    inline std::string get_template_name(const char* func_prefix, const char* func_name)
    {
        return std::string(func_prefix) + '_' + func_name + "_template";
    }

    // timing functions borrowed from rocblascommon/clients/include/utility.hpp
    double get_time_us();
    double get_time_us_sync(hipStream_t stream);
    double get_time_us_no_sync();

    // outputs bench logging
    template <typename T, typename... Ts>
    void log_bench(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        for(int i = 0; i < level - 1; i++)
            rocblas_cout << "    ";

        rocblas_cout << "rocsolver-bench -f " << func_name << " -r " << get_precision<T>() << ' ';
        print_pairs(rocblas_cout, " ", args...);
        rocblas_cout << '\n';
    }

    // outputs trace logging
    template <typename T, typename... Ts>
    void log_trace(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        for(int i = 0; i < level - 1; i++)
            rocblas_cout << "    ";

        rocblas_cout << get_template_name(func_prefix, func_name) << " (";
        print_pairs(rocblas_cout, ", ", args...);
        rocblas_cout << ')' << '\n';
    }

    // populates profile logging data with information from call_stack
    template <typename T>
    void log_profile(rocblas_handle handle, int level, const char* func_prefix, const char* func_name)
    {
        rocsolver_log_entry& from_stack = call_stack[handle][level - 1];
        rocsolver_log_entry& from_profile = profile[from_stack.name];

        hipStream_t stream;
        rocblas_get_stream(handle, &stream);
        double time = get_time_us_sync(stream) - from_stack.time;

        from_profile.name = from_stack.name;
        from_profile.calls++;
        from_profile.time += time;
        from_profile.internal_time += from_stack.internal_time;

        if(level > 1)
            call_stack[handle][level - 2].internal_time += time;
    }

public:
    // returns true if logging facilities are enabled
    inline bool is_logging_enabled()
    {
        return layer_mode > 0;
    }

    // logging function to be called upon entering a top-level (i.e. impl) function
    template <typename T, typename... Ts>
    void log_enter_top_level(rocblas_handle handle,
                             const char* func_prefix,
                             const char* func_name,
                             Ts... args)
    {
        int level = call_stack[handle].size();
        if(level != 0)
            ROCSOLVER_UNREACHABLE();

        rocblas_cout << "------- ENTER " << get_func_name<T>(func_prefix, func_name) << " -------"
                     << '\n';

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench<T>(level, func_prefix, func_name, args...);
    }

    // logging function to be called before exiting a top-level (i.e. impl) function
    template <typename T>
    void log_exit_top_level(rocblas_handle handle, const char* func_prefix, const char* func_name)
    {
        int level = call_stack[handle].size();
        if(level != 0)
            ROCSOLVER_UNREACHABLE();

        rocblas_cout << "-------  EXIT " << get_func_name<T>(func_prefix, func_name) << " -------"
                     << '\n'
                     << std::endl;

        call_stack.erase(handle);
    }

    // logging function to be called upon entering a sub-level (i.e. template) function
    template <typename T, typename... Ts>
    void log_enter(rocblas_handle handle, const char* func_prefix, const char* func_name, Ts... args)
    {
        rocsolver_log_entry entry;
        entry.name = get_template_name(func_prefix, func_name);
        entry.time = get_time_us_no_sync();
        call_stack[handle].push_back(entry);

        int level = call_stack[handle].size();

        if(layer_mode & rocblas_layer_mode_log_trace && level <= max_levels)
            log_trace<T>(level, func_prefix, func_name, args...);
    }

    // logging function to be called before exiting a sub-level (i.e. template) function
    template <typename T>
    void log_exit(rocblas_handle handle, const char* func_prefix, const char* func_name)
    {
        int level = call_stack[handle].size();

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile<T>(handle, level, func_prefix, func_name);

        call_stack[handle].pop_back();
    }

    friend rocblas_status rocsolver_logging_initialize(const rocblas_layer_mode layer_mode,
                                                       const rocblas_int max_levels);
    friend rocblas_status rocsolver_logging_cleanup(void);
};

extern rocsolver_logger* logger;
