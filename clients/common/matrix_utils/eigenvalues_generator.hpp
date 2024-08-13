/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "host_matrix.hpp"

namespace matxu
{

    // Generate eigenvalues following the patterns used in LAPACK's tests.
    //
    // See "A Testing Infrastructure for LAPACK's Symmetric Eigensolvers",
    // James W. Demmel, Osni A. Marques, Beresford N. Parlett, and Christof
    // Vomel; and references therein.
    //
    template<typename T, typename I>
    class GenerateEigenvalues
    {
        using S = decltype(std::real(T{}));

        template<typename K = S>
        auto operator()(I type, I n, K&& kappa = S(0)) -> HostMatrix<T, I>
        {
            auto eigenvalues = HostMatrix<T, I>::Zeros(1, n);
            if (kappa <= S(std::numeric_limits<S>::min()))
            {
                kappa = S(1) / std::sqrt(std::numeric_limits<S>::epsilon());
            }

            switch(type)
            {
                case 1:
                    if (n == 1)
                    {
                        eigenvalues(0) = S(1);
                    }
                    else
                    {
                        for (I i = 0; i < n; ++i)
                        {
                            if (i == 0)
                            {
                                eigenvalues(i) = S(1);
                            }
                            else
                            {
                                eigenvalues(i) = S(1)/kappa;
                            }
                        }
                    }
                    break;

                case 2:
                    if (n == 1)
                    {
                        eigenvalues(0) = S(1);
                    }
                    else
                    {
                        for (I i = 0; i < n; ++i)
                        {
                            if (i < n - 1)
                            {
                                eigenvalues(i) = S(1);
                            }
                            else
                            {
                                eigenvalues(i) = S(1)/kappa;
                            }
                        }
                    }
                    break;

                case 3:
                    if (n == 1)
                    {
                        eigenvalues(0) = S(1);
                    }
                    else
                    {
                        for (I i = 0; i < n; ++i)
                        {
                            eigenvalues(i) = std::pow(kappa, -i/(n - 1));
                        }
                    }
                    break;

                case 4:
                    if (n == 1)
                    {
                        eigenvalues(0) = S(1);
                    }
                    else
                    {
                        for (I i = 0; i < n; ++i)
                        {
                            eigenvalues(i) = S(1) -i/(n - S(1)) * (S(1) - S(1)/kappa);
                        }
                    }
                    break;

                case 5: // Not implemented: log-uniform numbers ranging in (1/kappa, 1)
                    eigenvalues = HostMatrix<T, I>::Empty();

                case 6: // Not implemented: random numbers
                    eigenvalues = HostMatrix<T, I>::Empty();

                case 7:
                    for (I i = 0; i < n; ++i)
                    {
                        if (i < n - 1)
                        {
                            eigenvalues(i) = std::numeric_limits<S>::epsilon() * i;
                        }
                        else
                        {
                            eigenvalues(i) = S(1);
                        }
                    }
                    break;

                case 8:
                    for (I i = 0; i < n; ++i)
                    {
                        if (i == 0)
                        {
                            eigenvalues(i) = S(1);
                        }
                        else
                        {
                            eigenvalues(i) = S(1) + std::sqrt(std::numeric_limits<S>::epsilon()) * i;
                        }
                    }
                    break;

                case 9:
                    for (I i = 0; i < n; ++i)
                    {
                        if (i == 0)
                        {
                            eigenvalues(i) = S(1);
                        }
                        else
                        {
                            eigenvalues(i) = eigenvalues(i - 1) + S(100) * i;
                        }
                    }
                    break;

                default:
                    eigenvalues = HostMatrix<T, I>::Empty();
                    break;
            }

            return eigenvalues;
        }
    };
} // namespace matxu
