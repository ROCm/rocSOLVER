/* **************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>

#include "common_host_helpers.hpp"

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

/***********************************************************************
 * template specializations                                            *
 ***********************************************************************/
template <>
constexpr auto rocblas_datatype_from_type<rocblas_half> = rocblas_datatype_f16_r;
template <>
constexpr auto rocblas_datatype_from_type<float> = rocblas_datatype_f32_r;
template <>
constexpr auto rocblas_datatype_from_type<double> = rocblas_datatype_f64_r;
template <>
constexpr auto rocblas_datatype_from_type<rocblas_float_complex> = rocblas_datatype_f32_c;
template <>
constexpr auto rocblas_datatype_from_type<rocblas_double_complex> = rocblas_datatype_f64_c;
template <>
constexpr auto rocblas_datatype_from_type<int8_t> = rocblas_datatype_i8_r;
template <>
constexpr auto rocblas_datatype_from_type<uint8_t> = rocblas_datatype_u8_r;
template <>
constexpr auto rocblas_datatype_from_type<int32_t> = rocblas_datatype_i32_r;
template <>
constexpr auto rocblas_datatype_from_type<uint32_t> = rocblas_datatype_u32_r;
template <>
constexpr auto rocblas_datatype_from_type<rocblas_bfloat16> = rocblas_datatype_bf16_r;
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
template <>
constexpr auto rocblas_datatype_from_type<rocblas_f8> = rocblas_datatype_f8_r;
template <>
constexpr auto rocblas_datatype_from_type<rocblas_bf8> = rocblas_datatype_bf8_r;
#endif

template <>
constexpr char rocblas_precision_string<rocblas_bfloat16>[] = "bf16_r";
template <>
constexpr char rocblas_precision_string<rocblas_half>[] = "f16_r";
template <>
constexpr char rocblas_precision_string<float>[] = "f32_r";
template <>
constexpr char rocblas_precision_string<double>[] = "f64_r";
template <>
constexpr char rocblas_precision_string<int8_t>[] = "i8_r";
template <>
constexpr char rocblas_precision_string<uint8_t>[] = "u8_r";
template <>
constexpr char rocblas_precision_string<int32_t>[] = "i32_r";
template <>
constexpr char rocblas_precision_string<uint32_t>[] = "u32_r";
template <>
constexpr char rocblas_precision_string<rocblas_float_complex>[] = "f32_c";
template <>
constexpr char rocblas_precision_string<rocblas_double_complex>[] = "f64_c";
#if 0 // Not implemented
template <> constexpr char rocblas_precision_string<rocblas_half_complex  >[] = "f16_c";
template <> constexpr char rocblas_precision_string<rocblas_i8_complex    >[] = "i8_c";
template <> constexpr char rocblas_precision_string<rocblas_u8_complex    >[] = "u8_c";
template <> constexpr char rocblas_precision_string<rocblas_i32_complex   >[] = "i32_c";
template <> constexpr char rocblas_precision_string<rocblas_u32_complex   >[] = "u32_c";
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
template <> constexpr char rocblas_precision_string<rocblas_f8            >[] = "f8_r";
template <> constexpr char rocblas_precision_string<rocblas_bf8           >[] = "bf8_r";
#endif
#endif

template <>
constexpr char rocblas2char_precision<float> = 's';
template <>
constexpr char rocblas2char_precision<double> = 'd';
template <>
constexpr char rocblas2char_precision<rocblas_float_complex> = 'c';
template <>
constexpr char rocblas2char_precision<rocblas_double_complex> = 'z';

/***********************************************************************
 * timing functions                                                    *
 ***********************************************************************/

/* CPU Timer (in microseconds): no GPU synchronization
 */
double get_time_us_no_sync()
{
    namespace sc = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/* CPU Timer (in microseconds): synchronize with the default device and return wall time
 */
double get_time_us()
{
    hipError_t status = hipDeviceSynchronize();
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}

/* CPU Timer (in microseconds): synchronize with given queue/stream and return wall time
 */
double get_time_us_sync(hipStream_t stream)
{
    hipError_t status = hipStreamSynchronize(stream);
#ifdef ROCSOLVER_LIBRARY
    if(status != hipSuccess)
        fmt::print(std::cerr, "{}: [{}] {}\n", __PRETTY_FUNCTION__, hipGetErrorName(status),
                   hipGetErrorString(status));
#else
    THROW_IF_HIP_ERROR(status);
#endif
    return get_time_us_no_sync();
}

/*! \brief Get warp size of the current device */
int get_device_warp_size()
{
    int warp_size;
    int device_id;
    hipError_t err;

    err = hipGetDevice(&device_id);

    if(err != hipSuccess)
    {
        return 0;
    }

    err = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device_id);

    if(err != hipSuccess)
    {
        return 0;
    }

    return warp_size;
}

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif
