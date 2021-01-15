/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocsolver_logger.hpp"
#include <sys/time.h>

rocsolver_logger* logger = nullptr;

/***************************************************************************
 * Logging set-up and tear-down
 ***************************************************************************/
extern "C" {

rocblas_status rocsolver_logging_initialize(const rocblas_layer_mode layer_mode,
                                            const rocblas_int max_levels)
{
    if(logger != nullptr)
        return rocblas_status_internal_error;
    if(max_levels < 1)
        return rocblas_status_invalid_value;

    logger = new rocsolver_logger();

    // set layer_mode from value of environment variable ROCSOLVER_LAYER
    const char* str_layer_mode = getenv("ROCSOLVER_LAYER");
    if(str_layer_mode)
        logger->layer_mode = static_cast<rocblas_layer_mode>(strtol(str_layer_mode, 0, 0));
    else
        logger->layer_mode = layer_mode;

    // set max_levels from value of environment variable ROCSOLVER_LEVELS
    const char* str_max_level = getenv("ROCSOLVER_LEVELS");
    if(str_max_level)
        logger->max_levels = static_cast<int>(strtol(str_max_level, 0, 0));
    else
        logger->max_levels = max_levels;

    return rocblas_status_success;
}

rocblas_status rocsolver_logging_cleanup()
{
    if(logger == nullptr)
        return rocblas_status_internal_error;

    // print profile logging results
    if(logger->layer_mode & rocblas_layer_mode_log_profile && logger->profile.size() > 0)
    {
        rocblas_cout << "------- PROFILE -------" << std::endl;
        for(auto it = logger->profile.begin(); it != logger->profile.end(); ++it)
        {
            rocblas_cout << it->first.c_str() << ": Calls: " << it->second.calls
                         << ", Total Time: " << it->second.time << " ms";
            if(it->second.internal_time > 0.0)
                rocblas_cout << " (in nested functions: " << it->second.internal_time << " ms)"
                             << std::endl;
            else
                rocblas_cout << std::endl;
        }
    }

    delete logger;
    logger = nullptr;

    return rocblas_status_success;
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
