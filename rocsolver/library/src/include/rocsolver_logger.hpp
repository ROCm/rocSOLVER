/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "common_host_helpers.hpp"
#include "rocsolver.h"
#include <iostream>
#include <string>
#include <unordered_map>

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

class rocsolver_logger
{
private:
    std::unordered_map<rocblas_handle, int> current_level;
    int max_levels;
    rocblas_layer_mode layer_mode;

    template <typename T>
    char get_precision();

    template <typename T>
    inline std::string get_func_name(const char* func_prefix, const char* func_name)
    {
        return std::string(func_prefix) + '_' + get_precision<T>() + func_name;
    }
    inline std::string get_template_name(const char* func_prefix, const char* func_name)
    {
        return std::string(func_prefix) + '_' + func_name + "_template";
    }

    template <typename T1, typename T2, typename... Ts>
    void log_arguments(std::ostream& os, const char* sep, T1 arg1, T2 arg2, Ts... args)
    {
        os << arg1 << ' ' << arg2;

        if(sizeof...(Ts) > 0)
        {
            os << sep;
            log_arguments(os, sep, args...);
        }
    }
    void log_arguments(std::ostream& os, const char* sep)
    {
        // do nothing
    }

    template <typename T, typename... Ts>
    void log_bench(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        for(int i = 0; i < level - 1; i++)
            std::cerr << "    ";

        std::cerr << "rocsolver-bench -f " << func_name << " -r " << get_precision<T>() << ' ';
        log_arguments(std::cerr, " ", args...);
        std::cerr << std::endl;
    }

    template <typename T, typename... Ts>
    void log_trace(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        for(int i = 0; i < level - 1; i++)
            std::cerr << "    ";

        std::cerr << get_template_name(func_prefix, func_name) << " (";
        log_arguments(std::cerr, ", ", args...);
        std::cerr << ')' << std::endl;
    }

    template <typename T>
    void log_profile(int level, const char* func_prefix, const char* func_name)
    {
    }

public:
    inline bool is_logging_enabled()
    {
        return layer_mode > 0;
    }

    template <typename T, typename... Ts>
    void log_enter_top_level(rocblas_handle handle,
                             const char* func_prefix,
                             const char* func_name,
                             Ts... args)
    {
        int level = current_level[handle];
        if(level != 0)
            ROCSOLVER_UNREACHABLE();

        std::cerr << "------- ENTER " << get_func_name<T>(func_prefix, func_name) << " -------"
                  << std::endl;

        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench<T>(level, func_prefix, func_name, args...);
    }

    template <typename T>
    void log_exit_top_level(rocblas_handle handle, const char* func_prefix, const char* func_name)
    {
        int level = current_level[handle];
        if(level != 0)
            ROCSOLVER_UNREACHABLE();

        std::cerr << "-------  EXIT " << get_func_name<T>(func_prefix, func_name) << " -------"
                  << std::endl
                  << std::endl;
    }

    template <typename T, typename... Ts>
    void log_enter(rocblas_handle handle, const char* func_prefix, const char* func_name, Ts... args)
    {
        int level = ++current_level[handle];

        if(layer_mode & rocblas_layer_mode_log_trace && level <= max_levels)
            log_trace<T>(level, func_prefix, func_name, args...);
    }

    template <typename T>
    void log_exit(rocblas_handle handle, const char* func_prefix, const char* func_name)
    {
        int level = current_level[handle]--;

        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile<T>(level, func_prefix, func_name);
    }

    friend rocblas_status rocsolver_logging_initialize(const rocblas_layer_mode layer_mode,
                                                       const rocblas_int max_levels);
    friend rocblas_status rocsolver_logging_cleanup(void);
};

extern rocsolver_logger* logger;
