
/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "clientcommon.hpp"

/*
 --------------------------------------------------
 Compute norm of difference between two matrices in
 Compressed Sparse Row (CSR) storage format
 --------------------------------------------------
 */

template <typename Iint, typename Ilong, typename T>
double csr_norm_error(char const norm_type,
                      char const uplo,
                      Iint const nrow,
                      Iint const ncol,

                      Ilong const* const Ap,
                      Iint const* const Ai,
                      T const* const Ax,

                      Ilong const* const Bp,
                      Iint const* const Bi,
                      T const* const Bx)
{
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);

    assert(Bp != nullptr);
    assert(Bi != nullptr);
    assert(Bx != nullptr);

    bool const is_frobenius = (norm_type == 'F') || (norm_type == 'f');

    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');
    bool const use_full = (!use_upper) && (!use_lower);

    T const zero = static_cast<T>(0);

    // ----------------------------
    // full length temporary vector
    // ----------------------------
    std::vector<T> v(ncol);
    for(size_t k = 0; k < v.size(); k++)
    {
        v[k] = zero;
    };

    double norm_err = static_cast<double>(0);

    for(Iint irow = 0; irow < nrow; irow++)
    {
        // ------------------------------------
        // copy data into temporary full vector
        // ------------------------------------
        Ilong const kstartA = Ap[irow];
        Ilong const kendA = Ap[irow + 1];
        Ilong const kstartB = Bp[irow];
        Ilong const kendB = Bp[irow + 1];

        for(Ilong k = kstartA; k < kendA; k++)
        {
            auto const colA = Ai[k];
            assert((0 <= colA) && (colA < ncol));

            bool const is_lower = (irow >= colA);
            bool const is_upper = (irow <= colA);

            bool do_assign = (use_lower && is_lower) || (use_upper && is_upper) || use_full;

            if(do_assign)
            {
                v[colA] = Ax[k];
            };
        };

        for(Ilong k = kstartB; k < kendB; k++)
        {
            Iint const colB = Bi[k];
            assert((0 <= colB) && (colB < ncol));

            bool const is_lower = (irow >= colB);
            bool const is_upper = (irow <= colB);
            bool const do_assign = (use_lower && is_lower) || (use_upper && is_upper) || use_full;
            if(do_assign)
            {
                v[colB] -= Bx[k];
            };
        };

        // ----------------------
        // evaluate norm of error
        // ----------------------
        for(Ilong k = kstartA; k < kendA; k++)
        {
            Iint const colA = Ai[k];

            double const vi = std::abs(v[colA]);
            v[colA] = zero;
            if(is_frobenius)
            {
                norm_err += vi * vi;
            }
            else
            {
                norm_err = std::max(norm_err, vi);
            };
        };

        for(Ilong k = kstartB; k < kendB; k++)
        {
            Iint const colB = Bi[k];

            double const vi = std::abs(v[colB]);
            v[colB] = zero;

            if(is_frobenius)
            {
                norm_err += vi * vi;
            }
            else
            {
                norm_err = std::max(norm_err, vi);
            };
        };
    };

    if(is_frobenius)
    {
        norm_err = sqrt(std::abs(norm_err));
    };

    return (norm_err);
}
