/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocblas.h"
#include "rocsolver_arguments.hpp"
#include <map>
#include <string>

#include "testing_bdsqr.hpp"
#include "testing_gebd2_gebrd.hpp"
#include "testing_gelq2_gelqf.hpp"
#include "testing_gels.hpp"
#include "testing_geql2_geqlf.hpp"
#include "testing_geqr2_geqrf.hpp"
#include "testing_gerq2_gerqf.hpp"
#include "testing_gesv.hpp"
#include "testing_gesvd.hpp"
#include "testing_getf2_getrf.hpp"
#include "testing_getf2_getrf_npvt.hpp"
#include "testing_getri.hpp"
#include "testing_getri_npvt.hpp"
#include "testing_getri_npvt_outofplace.hpp"
#include "testing_getri_outofplace.hpp"
#include "testing_getrs.hpp"
#include "testing_labrd.hpp"
#include "testing_lacgv.hpp"
#include "testing_larf.hpp"
#include "testing_larfb.hpp"
#include "testing_larfg.hpp"
#include "testing_larft.hpp"
#include "testing_laswp.hpp"
#include "testing_latrd.hpp"
#include "testing_orgbr_ungbr.hpp"
#include "testing_orglx_unglx.hpp"
#include "testing_orgtr_ungtr.hpp"
#include "testing_orgxl_ungxl.hpp"
#include "testing_orgxr_ungxr.hpp"
#include "testing_ormbr_unmbr.hpp"
#include "testing_ormlx_unmlx.hpp"
#include "testing_ormtr_unmtr.hpp"
#include "testing_ormxl_unmxl.hpp"
#include "testing_ormxr_unmxr.hpp"
#include "testing_posv.hpp"
#include "testing_potf2_potrf.hpp"
#include "testing_potri.hpp"
#include "testing_potrs.hpp"
#include "testing_stedc.hpp"
#include "testing_steqr.hpp"
#include "testing_sterf.hpp"
#include "testing_syev_heev.hpp"
#include "testing_syevd_heevd.hpp"
#include "testing_sygsx_hegsx.hpp"
#include "testing_sygv_hegv.hpp"
#include "testing_sygvd_hegvd.hpp"
#include "testing_sytxx_hetxx.hpp"
#include "testing_trtri.hpp"

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(Arguments&), str_less>;

// Function dispatcher for rocSOLVER tests
class rocsolver_dispatcher
{
    template <typename T>
    static rocblas_status run_function(const char* name, Arguments& argus)
    {
        // Map for functions that support all precisions
        static const func_map map = {
            {"laswp", testing_laswp<T>},
            {"larfg", testing_larfg<T>},
            {"larf", testing_larf<T>},
            {"larft", testing_larft<T>},
            {"larfb", testing_larfb<T>},
            {"latrd", testing_latrd<T>},
            {"labrd", testing_labrd<T>},
            {"bdsqr", testing_bdsqr<T>},
            {"steqr", testing_steqr<T>},
            {"stedc", testing_stedc<T>},
            // potrf
            {"potf2", testing_potf2_potrf<false, false, 0, T>},
            {"potf2_batched", testing_potf2_potrf<true, true, 0, T>},
            {"potf2_strided_batched", testing_potf2_potrf<false, true, 0, T>},
            {"potrf", testing_potf2_potrf<false, false, 1, T>},
            {"potrf_batched", testing_potf2_potrf<true, true, 1, T>},
            {"potrf_strided_batched", testing_potf2_potrf<false, true, 1, T>},
            // potrs
            {"potrs", testing_potrs<false, false, T>},
            {"potrs_batched", testing_potrs<true, true, T>},
            {"potrs_strided_batched", testing_potrs<false, true, T>},
            // posv
            {"posv", testing_posv<false, false, T>},
            {"posv_batched", testing_posv<true, true, T>},
            {"posv_strided_batched", testing_posv<false, true, T>},
            // potri
            {"potri", testing_potri<false, false, T>},
            {"potri_batched", testing_potri<true, true, T>},
            {"potri_strided_batched", testing_potri<false, true, T>},
            // getrf_npvt
            {"getf2_npvt", testing_getf2_getrf_npvt<false, false, 0, T>},
            {"getf2_npvt_batched", testing_getf2_getrf_npvt<true, true, 0, T>},
            {"getf2_npvt_strided_batched", testing_getf2_getrf_npvt<false, true, 0, T>},
            {"getrf_npvt", testing_getf2_getrf_npvt<false, false, 1, T>},
            {"getrf_npvt_batched", testing_getf2_getrf_npvt<true, true, 1, T>},
            {"getrf_npvt_strided_batched", testing_getf2_getrf_npvt<false, true, 1, T>},
            // getrf
            {"getf2", testing_getf2_getrf<false, false, 0, T>},
            {"getf2_batched", testing_getf2_getrf<true, true, 0, T>},
            {"getf2_strided_batched", testing_getf2_getrf<false, true, 0, T>},
            {"getrf", testing_getf2_getrf<false, false, 1, T>},
            {"getrf_batched", testing_getf2_getrf<true, true, 1, T>},
            {"getrf_strided_batched", testing_getf2_getrf<false, true, 1, T>},
            // geqrf
            {"geqr2", testing_geqr2_geqrf<false, false, 0, T>},
            {"geqr2_batched", testing_geqr2_geqrf<true, true, 0, T>},
            {"geqr2_strided_batched", testing_geqr2_geqrf<false, true, 0, T>},
            {"geqrf", testing_geqr2_geqrf<false, false, 1, T>},
            {"geqrf_batched", testing_geqr2_geqrf<true, true, 1, T>},
            {"geqrf_strided_batched", testing_geqr2_geqrf<false, true, 1, T>},
            {"geqrf_ptr_batched", testing_geqr2_geqrf<true, false, 1, T>},
            // gerqf
            {"gerq2", testing_gerq2_gerqf<false, false, 0, T>},
            {"gerq2_batched", testing_gerq2_gerqf<true, true, 0, T>},
            {"gerq2_strided_batched", testing_gerq2_gerqf<false, true, 0, T>},
            {"gerqf", testing_gerq2_gerqf<false, false, 1, T>},
            {"gerqf_batched", testing_gerq2_gerqf<true, true, 1, T>},
            {"gerqf_strided_batched", testing_gerq2_gerqf<false, true, 1, T>},
            // geqlf
            {"geql2", testing_geql2_geqlf<false, false, 0, T>},
            {"geql2_batched", testing_geql2_geqlf<true, true, 0, T>},
            {"geql2_strided_batched", testing_geql2_geqlf<false, true, 0, T>},
            {"geqlf", testing_geql2_geqlf<false, false, 1, T>},
            {"geqlf_batched", testing_geql2_geqlf<true, true, 1, T>},
            {"geqlf_strided_batched", testing_geql2_geqlf<false, true, 1, T>},
            // gelqf
            {"gelq2", testing_gelq2_gelqf<false, false, 0, T>},
            {"gelq2_batched", testing_gelq2_gelqf<true, true, 0, T>},
            {"gelq2_strided_batched", testing_gelq2_gelqf<false, true, 0, T>},
            {"gelqf", testing_gelq2_gelqf<false, false, 1, T>},
            {"gelqf_batched", testing_gelq2_gelqf<true, true, 1, T>},
            {"gelqf_strided_batched", testing_gelq2_gelqf<false, true, 1, T>},
            // getrs
            {"getrs", testing_getrs<false, false, T>},
            {"getrs_batched", testing_getrs<true, true, T>},
            {"getrs_strided_batched", testing_getrs<false, true, T>},
            // gesv
            {"gesv", testing_gesv<false, false, T>},
            {"gesv_batched", testing_gesv<true, true, T>},
            {"gesv_strided_batched", testing_gesv<false, true, T>},
            // gesvd
            {"gesvd", testing_gesvd<false, false, T>},
            {"gesvd_batched", testing_gesvd<true, true, T>},
            {"gesvd_strided_batched", testing_gesvd<false, true, T>},
            // trtri
            {"trtri", testing_trtri<false, false, T>},
            {"trtri_batched", testing_trtri<true, true, T>},
            {"trtri_strided_batched", testing_trtri<false, true, T>},
            // getri
            {"getri", testing_getri<false, false, T>},
            {"getri_batched", testing_getri<true, true, T>},
            {"getri_strided_batched", testing_getri<false, true, T>},
            // getri_npvt
            {"getri_npvt", testing_getri_npvt<false, false, T>},
            {"getri_npvt_batched", testing_getri_npvt<true, true, T>},
            {"getri_npvt_strided_batched", testing_getri_npvt<false, true, T>},
            // getri_outofplace
            {"getri_outofplace", testing_getri_outofplace<false, false, T>},
            {"getri_outofplace_batched", testing_getri_outofplace<true, true, T>},
            {"getri_outofplace_strided_batched", testing_getri_outofplace<false, true, T>},
            // getri_npvt_outofplace
            {"getri_npvt_outofplace", testing_getri_npvt_outofplace<false, false, T>},
            {"getri_npvt_outofplace_batched", testing_getri_npvt_outofplace<true, true, T>},
            {"getri_npvt_outofplace_strided_batched", testing_getri_npvt_outofplace<false, true, T>},
            // gels
            {"gels", testing_gels<false, false, T>},
            {"gels_batched", testing_gels<true, true, T>},
            {"gels_strided_batched", testing_gels<false, true, T>},
            // gebrd
            {"gebd2", testing_gebd2_gebrd<false, false, 0, T>},
            {"gebd2_batched", testing_gebd2_gebrd<true, true, 0, T>},
            {"gebd2_strided_batched", testing_gebd2_gebrd<false, true, 0, T>},
            {"gebrd", testing_gebd2_gebrd<false, false, 1, T>},
            {"gebrd_batched", testing_gebd2_gebrd<true, true, 1, T>},
            {"gebrd_strided_batched", testing_gebd2_gebrd<false, true, 1, T>},
        };

        // Grab function from the map and execute
        auto match = map.find(name);
        if(match != map.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
    }

    template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
    static rocblas_status run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support only single and double precisions
        static const func_map map_real = {
            {"sterf", testing_sterf<T>},
            // orgxx
            {"org2r", testing_orgxr_ungxr<T, 0>},
            {"orgqr", testing_orgxr_ungxr<T, 1>},
            {"org2l", testing_orgxl_ungxl<T, 0>},
            {"orgql", testing_orgxl_ungxl<T, 1>},
            {"orgl2", testing_orglx_unglx<T, 0>},
            {"orglq", testing_orglx_unglx<T, 1>},
            {"orgbr", testing_orgbr_ungbr<T>},
            {"orgtr", testing_orgtr_ungtr<T>},
            // ormxx
            {"orm2r", testing_ormxr_unmxr<T, 0>},
            {"ormqr", testing_ormxr_unmxr<T, 1>},
            {"orm2l", testing_ormxl_unmxl<T, 0>},
            {"ormql", testing_ormxl_unmxl<T, 1>},
            {"orml2", testing_ormlx_unmlx<T, 0>},
            {"ormlq", testing_ormlx_unmlx<T, 1>},
            {"ormbr", testing_ormbr_unmbr<T>},
            {"ormtr", testing_ormtr_unmtr<T>},
            // sytrd
            {"sytd2", testing_sytxx_hetxx<false, false, 0, T>},
            {"sytd2_batched", testing_sytxx_hetxx<true, true, 0, T>},
            {"sytd2_strided_batched", testing_sytxx_hetxx<false, true, 0, T>},
            {"sytrd", testing_sytxx_hetxx<false, false, 1, T>},
            {"sytrd_batched", testing_sytxx_hetxx<true, true, 1, T>},
            {"sytrd_strided_batched", testing_sytxx_hetxx<false, true, 1, T>},
            // sygst
            {"sygs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"sygs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"sygs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"sygst", testing_sygsx_hegsx<false, false, 1, T>},
            {"sygst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"sygst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
            // syev
            {"syev", testing_syev_heev<false, false, T>},
            {"syev_batched", testing_syev_heev<true, true, T>},
            {"syev_strided_batched", testing_syev_heev<false, true, T>},
            // syevd
            {"syevd", testing_syevd_heevd<false, false, T>},
            {"syevd_batched", testing_syevd_heevd<true, true, T>},
            {"syevd_strided_batched", testing_syevd_heevd<false, true, T>},
            // sygv
            {"sygv", testing_sygv_hegv<false, false, T>},
            {"sygv_batched", testing_sygv_hegv<true, true, T>},
            {"sygv_strided_batched", testing_sygv_hegv<false, true, T>},
            // sygvd
            {"sygvd", testing_sygvd_hegvd<false, false, T>},
            {"sygvd_batched", testing_sygvd_hegvd<true, true, T>},
            {"sygvd_strided_batched", testing_sygvd_hegvd<false, true, T>},
        };

        // Grab function from the map and execute
        auto match = map_real.find(name);
        if(match != map_real.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
    }

    template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
    static rocblas_status run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support only single-complex and double-complex precisions
        static const func_map map_complex = {
            {"lacgv", testing_lacgv<T>},
            // ungxx
            {"ung2r", testing_orgxr_ungxr<T, 0>},
            {"ungqr", testing_orgxr_ungxr<T, 1>},
            {"ung2l", testing_orgxl_ungxl<T, 0>},
            {"ungql", testing_orgxl_ungxl<T, 1>},
            {"ungl2", testing_orglx_unglx<T, 0>},
            {"unglq", testing_orglx_unglx<T, 1>},
            {"ungbr", testing_orgbr_ungbr<T>},
            {"ungtr", testing_orgtr_ungtr<T>},
            // unmxx
            {"unm2r", testing_ormxr_unmxr<T, 0>},
            {"unmqr", testing_ormxr_unmxr<T, 1>},
            {"unm2l", testing_ormxl_unmxl<T, 0>},
            {"unmql", testing_ormxl_unmxl<T, 1>},
            {"unml2", testing_ormlx_unmlx<T, 0>},
            {"unmlq", testing_ormlx_unmlx<T, 1>},
            {"unmbr", testing_ormbr_unmbr<T>},
            {"unmtr", testing_ormtr_unmtr<T>},
            // hetrd
            {"hetd2", testing_sytxx_hetxx<false, false, 0, T>},
            {"hetd2_batched", testing_sytxx_hetxx<true, true, 0, T>},
            {"hetd2_strided_batched", testing_sytxx_hetxx<false, true, 0, T>},
            {"hetrd", testing_sytxx_hetxx<false, false, 1, T>},
            {"hetrd_batched", testing_sytxx_hetxx<true, true, 1, T>},
            {"hetrd_strided_batched", testing_sytxx_hetxx<false, true, 1, T>},
            // hegst
            {"hegs2", testing_sygsx_hegsx<false, false, 0, T>},
            {"hegs2_batched", testing_sygsx_hegsx<true, true, 0, T>},
            {"hegs2_strided_batched", testing_sygsx_hegsx<false, true, 0, T>},
            {"hegst", testing_sygsx_hegsx<false, false, 1, T>},
            {"hegst_batched", testing_sygsx_hegsx<true, true, 1, T>},
            {"hegst_strided_batched", testing_sygsx_hegsx<false, true, 1, T>},
            // heev
            {"heev", testing_syev_heev<false, false, T>},
            {"heev_batched", testing_syev_heev<true, true, T>},
            {"heev_strided_batched", testing_syev_heev<false, true, T>},
            // heevd
            {"heevd", testing_syevd_heevd<false, false, T>},
            {"heevd_batched", testing_syevd_heevd<true, true, T>},
            {"heevd_strided_batched", testing_syevd_heevd<false, true, T>},
            // hegv
            {"hegv", testing_sygv_hegv<false, false, T>},
            {"hegv_batched", testing_sygv_hegv<true, true, T>},
            {"hegv_strided_batched", testing_sygv_hegv<false, true, T>},
            // hegvd
            {"hegvd", testing_sygvd_hegvd<false, false, T>},
            {"hegvd_batched", testing_sygvd_hegvd<true, true, T>},
            {"hegvd_strided_batched", testing_sygvd_hegvd<false, true, T>},
        };

        // Grab function from the map and execute
        auto match = map_complex.find(name);
        if(match != map_complex.end())
        {
            match->second(argus);
            return rocblas_status_success;
        }
        else
            return rocblas_status_invalid_value;
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
            if(precision == 's')
                status = run_function_limited_precision<float>(name.c_str(), argus);
            else if(precision == 'd')
                status = run_function_limited_precision<double>(name.c_str(), argus);
            else if(precision == 'c')
                status = run_function_limited_precision<rocblas_float_complex>(name.c_str(), argus);
            else if(precision == 'z')
                status = run_function_limited_precision<rocblas_double_complex>(name.c_str(), argus);
        }

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
