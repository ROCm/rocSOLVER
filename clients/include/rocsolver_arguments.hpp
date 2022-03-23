/* ************************************************************************
 * Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <rocblas.h>

#include "rocblascommon/program_options.hpp"

using variables_map = roc::variables_map;
using variable_value = roc::variable_value;

class Arguments : private std::map<std::string, variable_value>
{
    using base = std::map<std::string, variable_value>;

    // names of arguments that have not yet been used by tests
    std::set<std::string> to_consume;

public:
    // test options
    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int timing = 0;
    rocblas_int perf = 0;
    rocblas_int singular = 0;
    rocblas_int iters = 5;
    rocblas_int mem_query = 0;
    rocblas_int profile = 0;
    rocblas_int batch_count = 1;

    // get and set function arguments
    template <typename T>
    const T& peek(const std::string& name) const
    {
        return at(name).as<T>();
    }

    template <typename T>
    const T& get(const std::string& name)
    {
        to_consume.erase(name);
        auto val = find(name);
        if(val != end() && !val->second.empty())
            return val->second.as<T>();
        else
            throw std::invalid_argument("No value provided for " + name);
    }

    template <typename T>
    const T get(const std::string& name, const T& default_value)
    {
        to_consume.erase(name);
        auto val = find(name);
        if(val != end() && !val->second.empty() && !val->second.defaulted())
            return val->second.as<T>();
        else
            return default_value;
    }

    template <typename T>
    void set(const std::string& name, const T& val)
    {
        to_consume.insert(name);
        base::operator[](name) = variable_value(val, false);
    }

    void populate(const variables_map& vm)
    {
        for(auto& pair : vm)
        {
            base::operator[](pair.first) = pair.second;

            if(!pair.second.empty() && !pair.second.defaulted())
                to_consume.insert(pair.first);
        }

        // remove test arguments
        to_consume.erase("help");
        to_consume.erase("function");
        to_consume.erase("precision");
        to_consume.erase("batch_count");
        to_consume.erase("verify");
        to_consume.erase("iters");
        to_consume.erase("mem_query");
        to_consume.erase("profile");
        to_consume.erase("perf");
        to_consume.erase("singular");
        to_consume.erase("device");
    }

    void clear()
    {
        to_consume.clear();
        base::clear();
    }

    // validate function arguments
    void validate_precision(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char precision = val->second.as<char>();
        if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_operation(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char trans = val->second.as<char>();
        if(trans != 'N' && trans != 'T' && trans != 'C')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_side(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char side = val->second.as<char>();
        if(side != 'L' && side != 'R' && side != 'B')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_fill(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char uplo = val->second.as<char>();
        if(uplo != 'U' && uplo != 'L' && uplo != 'F')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_diag(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char diag = val->second.as<char>();
        if(diag != 'N' && diag != 'U')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_direct(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char direct = val->second.as<char>();
        if(direct != 'F' && direct != 'B')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_storev(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char storev = val->second.as<char>();
        if(storev != 'R' && storev != 'C')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_svect(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char svect = val->second.as<char>();
        if(svect != 'A' && svect != 'S' && svect != 'O' && svect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_workmode(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char workmode = val->second.as<char>();
        if(workmode != 'O' && workmode != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_evect(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char evect = val->second.as<char>();
        if(evect != 'V' && evect != 'I' && evect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_erange(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char range = val->second.as<char>();
        if(range != 'A' && range != 'V' && range != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_eorder(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char order = val->second.as<char>();
        if(order != 'B' && order != 'E')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_itype(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char itype = val->second.as<char>();
        if(itype != '1' && itype != '2' && itype != '3')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_consumed() const
    {
        if(!to_consume.empty())
            throw std::invalid_argument(
                fmt::format("Not all arguments were consumed: {}", fmt::join(to_consume, " ")));
    }
};
