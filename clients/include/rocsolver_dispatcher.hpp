/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocsolver_arguments.hpp"
#include <map>
#include <string>

#include "testing_gesvd.hpp"
#include "testing_sygsx_hegsx.hpp"

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(Arguments), str_less>;

// Function dispatcher for rocSOLVER tests
class rocsolver_dispatcher
{
    template <typename T>
    static rocblas_status run_function(const char* name, Arguments argus)
    {
        // Map for functions that support all precisions
        static const func_map map = {
            {"gesvd", testing_gesvd<false, false, T>},
            {"gesvd_batched", testing_gesvd<true, true, T>},
            {"gesvd_strided_batched", testing_gesvd<false, true, T>},
        };

        // Map for functions that support single and double precisions
        static const func_map map_real = {
            {"sygs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"sygs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"sygs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"sygst", testing_sygsx_hegsx<false, false, 1, T>},
            {"sygst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"sygst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
        };

        // Map for functions that support single complex and double complex precisions
        static const func_map map_complex = {
            {"hegs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"hegs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"hegs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"hegst", testing_sygsx_hegsx<false, false, 1, T>},
            {"hegst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"hegst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
        };

        // Grab function from the map and execute
        auto match = map.find(name);
        if(match != map.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
        {
            if(!is_complex<T>)
            {
                match = map_real.find(name);
                if(match == map_real.end())
                    return rocblas_status_invalid_value;
            }
            else
            {
                match = map_complex.find(name);
                if(match == map_complex.end())
                    return rocblas_status_invalid_value;
            }

            match->second(argus);
            return rocblas_status_success;
        }
    }

public:
    static void invoke(const std::string& name, char precision, Arguments& argus)
    {
        rocblas_status status;

        if(precision == 's')
            status = run_function<float>(name.c_str(), argus);
        else if(precision == 'd')
            status = run_function<double>(name.c_str(), argus);
        else if(precision == 'c')
            status = run_function<rocblas_float_complex>(name.c_str(), argus);
        else if(precision == 'z')
            status = run_function<rocblas_double_complex>(name.c_str(), argus);
        else
            throw std::invalid_argument("Invalid value for --precision");

        if(status == rocblas_status_invalid_value)
        {
            std::string msg = "Invalid combination --function ";
            msg += name;
            msg += " --precision ";
            msg += precision;
            throw std::invalid_argument(msg);
        }
    }
};
