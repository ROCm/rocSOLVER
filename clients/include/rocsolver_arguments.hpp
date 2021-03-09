/* ************************************************************************
 * Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocblascommon/program_options.hpp"
#include "rocsolver_ostream.hpp"
#include <set>
#include <sstream>

class Arguments : private std::map<std::string, variable_value>
{
    using base = std::map<std::string, variable_value>;

    std::set<std::string> to_consume;

public:
    ~Arguments()
    {
        if(to_consume.size() > 0)
        {
            std::stringstream ss;
            ss << "WARNING: Not all arguments were consumed:";
            for(std::string name : to_consume)
                ss << ' ' << name;
            rocsolver_cerr << ss.str() << std::endl;
        }
    }

    // test options
    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int timing = 0;
    rocblas_int perf = 0;
    rocblas_int singular = 0;
    rocblas_int iters = 5;
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
        return at(name).as<T>();
    }

    template <typename T>
    const T& get(const std::string& name, const T& default_value)
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
        to_consume.erase("perf");
        to_consume.erase("singular");
        to_consume.erase("device");
    }

    // validate function arguments
    void validate_fill(const std::string name)
    {
        char uplo = at(name).as<char>();
        if(uplo != 'U' && uplo != 'L' && uplo != 'F')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_svect(const std::string name)
    {
        char svect = at(name).as<char>();
        if(svect != 'A' && svect != 'S' && svect != 'O' && svect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_workmode(const std::string name)
    {
        char workmode = at(name).as<char>();
        if(workmode != 'O' && workmode != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_evect(const std::string name)
    {
        char evect = at(name).as<char>();
        if(evect != 'V' && evect != 'I' && evect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_itype(const std::string name)
    {
        char itype = at(name).as<char>();
        if(itype != '1' && itype != '2' && itype != '3')
            throw std::invalid_argument("Invalid value for " + name);
    }

    // TODO: Remove these fields
    rocblas_int M = 128;
    rocblas_int N = 128;
    rocblas_int K = 128;
    rocblas_int S4 = 128;
    rocblas_int k1 = 1;
    rocblas_int k2 = 2;

    rocblas_int lda = 128;
    rocblas_int ldb = 128;
    rocblas_int ldc = 128;
    rocblas_int ldv = 128;
    rocblas_int ldt = 128;

    rocblas_int incx = 1;
    rocblas_int incy = 1;
    rocblas_int incd = 1;
    rocblas_int incb = 1;

    double alpha = 1.0;
    double beta = 0.0;

    char transA_option = 'N';
    char transB_option = 'N';
    char transH_option = 'N';
    char side_option = 'L';
    char uplo_option = 'L';
    char diag_option = 'N';
    char direct_option = 'F';
    char storev = 'C';
    char evect = 'N';

    rocblas_int bsa = 128 * 128;
    rocblas_int bsb = 128 * 128;
    rocblas_int bsc = 128 * 128;
    rocblas_int bsp = 128;

    char workmode = 'O';
    char itype = '1';
};
