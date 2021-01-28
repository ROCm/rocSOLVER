/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "common_host_helpers.hpp"
#include "common_ostream_helpers.hpp"
#include "rocsolver.h"
#include <memory>
#include <mutex>
#include <unordered_map>

/***************************************************************************
 * rocSOLVER logging macros
 ***************************************************************************/

#define ROCSOLVER_ENTER_TOP(name, ...)                                                      \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                           \
    do                                                                                      \
    {                                                                                       \
        if(rocsolver_logger::is_logging_enabled())                                          \
        {                                                                                   \
            rocsolver_logger::instance()->log_enter_top_level<T>(handle, "rocsolver", name, \
                                                                 __VA_ARGS__);              \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(true, handle);  \
        }                                                                                   \
    } while(0)
#define ROCSOLVER_ENTER(name, ...)                                                              \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                               \
    do                                                                                          \
    {                                                                                           \
        if(rocsolver_logger::is_logging_enabled())                                              \
        {                                                                                       \
            rocsolver_logger::instance()->log_enter<T>(handle, "rocsolver", name, __VA_ARGS__); \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(false, handle);     \
        }                                                                                       \
    } while(0)
#define ROCBLAS_ENTER(name, ...)                                                              \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                             \
    do                                                                                        \
    {                                                                                         \
        if(rocsolver_logger::is_logging_enabled())                                            \
        {                                                                                     \
            rocsolver_logger::instance()->log_enter<T>(handle, "rocblas", name, __VA_ARGS__); \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(false, handle);   \
        }                                                                                     \
    } while(0)

/***************************************************************************
 * The rocsolver_log_entry struct records function data for profile logging
 * purposes.
 ***************************************************************************/
struct rocsolver_log_entry
{
    std::string name;
    int level;
    int calls;
    double time;
    double internal_time;

    rocsolver_log_entry()
        : level(0)
        , calls(0)
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
    // static singleton instance
    static rocsolver_logger* _instance;
    // static mutex for multithreading
    static std::mutex _mutex;
    // profile logging data keyed by function name
    std::unordered_map<std::string, rocsolver_log_entry> profile;
    // function call stack keyed by handle
    std::unordered_map<rocblas_handle, std::vector<rocsolver_log_entry>> call_stack;
    // the maximum depth at which nested function calls will appear in the log
    int max_levels;
    // layer mode enum describing which logging facilities are enabled
    rocblas_layer_mode layer_mode;
    // streams for different logging types
    std::unique_ptr<rocsolver_ostream> trace_os;
    std::unique_ptr<rocsolver_ostream> bench_os;
    std::unique_ptr<rocsolver_ostream> profile_os;

    // returns a unique_ptr to a file stream or a given default stream
    auto open_log_stream(const char* environment_variable_name);

    // returns a log entry on the call stack
    rocsolver_log_entry& push_log_entry(rocblas_handle handle, std::string name);
    rocsolver_log_entry& peek_log_entry(rocblas_handle handle);
    rocsolver_log_entry pop_log_entry(rocblas_handle handle);

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
            *bench_os << "    ";

        *bench_os << "./rocsolver-bench -f " << func_name << " -r " << get_precision<T>() << ' ';
        print_pairs(*bench_os, " ", args...);
        *bench_os << std::endl;
    }

    // outputs trace logging
    template <typename T, typename... Ts>
    void log_trace(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        for(int i = 0; i < level - 1; i++)
            *trace_os << "    ";

        *trace_os << get_template_name(func_prefix, func_name) << " (";
        print_pairs(*trace_os, ", ", args...);
        *trace_os << ')' << '\n';
    }

    // populates profile logging data with information from call_stack
    template <typename T>
    void log_profile(rocblas_handle handle, rocsolver_log_entry& from_stack)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);
        double time = get_time_us_sync(stream) - from_stack.time;

        rocsolver_logger::_mutex.lock();
        rocsolver_log_entry& from_profile = profile[from_stack.name];
        from_profile.name = from_stack.name;
        from_profile.calls++;
        from_profile.time += time;
        from_profile.internal_time += from_stack.internal_time;

        if(from_stack.level > 1)
            peek_log_entry(handle).internal_time += time;
        rocsolver_logger::_mutex.unlock();
    }

public:
    // return the singleton instance
    static inline rocsolver_logger* instance()
    {
        return rocsolver_logger::_instance;
    }

    // returns true if logging facilities are enabled
    static inline bool is_logging_enabled()
    {
        return (rocsolver_logger::_instance != nullptr)
            && (rocsolver_logger::_instance->layer_mode
                & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                   | rocblas_layer_mode_log_profile));
    }

    // logging function to be called upon entering a top-level (i.e. impl) function
    template <typename T, typename... Ts>
    void log_enter_top_level(rocblas_handle handle,
                             const char* func_prefix,
                             const char* func_name,
                             Ts... args)
    {
        rocsolver_logger::_mutex.lock();
        auto entry = push_log_entry(handle, get_func_name<T>(func_prefix, func_name));
        rocsolver_logger::_mutex.unlock();
        ROCSOLVER_ASSUME(entry.level == 0);

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench<T>(entry.level, func_prefix, func_name, args...);

        if(trace_os)
            *trace_os << "------- ENTER " << entry.name << " -------\n";
    }

    // logging function to be called before exiting a top-level (i.e. impl) function
    template <typename T>
    void log_exit_top_level(rocblas_handle handle)
    {
        rocsolver_logger::_mutex.lock();
        auto entry = pop_log_entry(handle);
        rocsolver_logger::_mutex.unlock();
        ROCSOLVER_ASSUME(entry.level == 0);

        if(trace_os)
            *trace_os << "-------  EXIT " << entry.name << " -------\n" << std::endl;
    }

    // logging function to be called upon entering a sub-level (i.e. template) function
    template <typename T, typename... Ts>
    void log_enter(rocblas_handle handle, const char* func_prefix, const char* func_name, Ts... args)
    {
        rocsolver_logger::_mutex.lock();
        auto entry = push_log_entry(handle, get_template_name(func_prefix, func_name));
        rocsolver_logger::_mutex.unlock();

        if(layer_mode & rocblas_layer_mode_log_trace && entry.level <= max_levels)
            log_trace<T>(entry.level, func_prefix, func_name, args...);
    }

    // logging function to be called before exiting a sub-level (i.e. template) function
    template <typename T>
    void log_exit(rocblas_handle handle)
    {
        rocsolver_logger::_mutex.lock();
        auto entry = pop_log_entry(handle);
        rocsolver_logger::_mutex.unlock();

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile<T>(handle, entry);
    }

    /***************************************************************************
     * The scope_guard struct will call an appropriate logging exit function
     * upon the function losing scope.
     ***************************************************************************/
    template <typename T>
    struct scope_guard
    {
        bool top_level;
        rocblas_handle handle;

        scope_guard(bool top_level, rocblas_handle handle)
            : top_level(top_level)
            , handle(handle)
        {
        }
        ~scope_guard()
        {
            if(top_level)
                rocsolver_logger::instance()->log_exit_top_level<T>(handle);
            else
                rocsolver_logger::instance()->log_exit<T>(handle);
        }
    };

    friend rocblas_status rocsolver_logging_initialize(const rocblas_layer_mode layer_mode,
                                                       const rocblas_int max_levels);
    friend rocblas_status rocsolver_logging_cleanup(void);
};
