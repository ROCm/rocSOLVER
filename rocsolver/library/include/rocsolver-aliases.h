/* ************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//
//
// IMPORTANT:
//
// THESE ALIASES ARE NOT MAINTAINED ANYMORE. THEY EXIST ONLY FOR BACKWARDS
// COMPATIBILITY. USE ROCBLAS TYPES AND FUNCTIONS DIRECTLY.
//
//

#ifndef ROCSOLVER_ALIASES_H_
#define ROCSOLVER_ALIASES_H_

#include "rocsolver-export.h"
#include "rocsolver-extra-types.h"
#include <rocblas.h>

// rocblas original types

/*! \deprecated Use \c rocblas_int.
 */
typedef rocblas_int rocsolver_int __attribute__((deprecated("use rocblas_int")));

/*! \deprecated Use \c rocblas_stride.
 */
typedef rocblas_stride rocsolver_stride __attribute__((deprecated("use rocblas_stride")));

/*! \deprecated Use \c rocblas_float_complex.
 */
typedef rocblas_float_complex rocsolver_float_complex
    __attribute__((deprecated("use rocblas_float_complex")));

/*! \deprecated Use \c rocblas_double_complex.
 */
typedef rocblas_double_complex rocsolver_double_complex
    __attribute__((deprecated("use rocblas_double_complex")));

/*! \deprecated Use \c rocblas_half.
 */
typedef rocblas_half rocsolver_half __attribute__((deprecated("use rocblas_half")));

/*! \deprecated Use \c rocblas_handle.
 */
typedef rocblas_handle rocsolver_handle __attribute__((deprecated("use rocblas_handle")));

/*! \deprecated Use \c rocblas_operation.
 */
typedef rocblas_operation rocsolver_operation __attribute__((deprecated("use rocblas_operation")));

/*! \deprecated Use \c rocblas_fill.
 */
typedef rocblas_fill rocsolver_fill __attribute__((deprecated("use rocblas_fill")));

/*! \deprecated Use \c rocblas_diagonal.
 */
typedef rocblas_diagonal rocsolver_diagonal __attribute__((deprecated("use rocblas_diagonal")));

/*! \deprecated Use \c rocblas_stide.
 */
typedef rocblas_side rocsolver_side __attribute__((deprecated("use rocblas_side")));

/*! \deprecated Use \c rocblas_status.
 */
typedef rocblas_status rocsolver_status __attribute__((deprecated("use rocblas_status")));

/*! \deprecated Use \c rocblas_layer_mode.
 */
typedef rocblas_layer_mode rocsolver_layer_mode
    __attribute__((deprecated("use rocblas_layer_mode")));

// extras types only used in rocsolver

/*! \deprecated Use \c rocblas_direct
 */
typedef rocblas_direct rocsolver_direction __attribute__((deprecated("use rocblas_direct")));

/*! \deprecated Use \c rocblas_storev.
 */
typedef rocblas_storev rocsolver_storev __attribute__((deprecated("use rocblas_storev")));

// auxiliaries
#ifdef __cplusplus
extern "C" {
#endif

// deprecated functions use deprecated types
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/*! \brief Creates a handle and sets the pointer mode to \c
   rocblas_pointer_mode_device. \deprecated Use \c rocblas_create_handle.
*/
ROCSOLVER_EXPORT rocsolver_status rocsolver_create_handle(rocsolver_handle* handle)
    __attribute__((deprecated("use rocblas_create_handle")));

/*! \deprecated Use \c rocblas_destroy_handle.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_destroy_handle(rocsolver_handle handle)
    __attribute__((deprecated("use rocblas_destroy_handle")));

// rocblas_add_stream was removed in ROCm 3.6; use rocblas_set_stream
//
// ROCSOLVER_EXPORT rocsolver_status
// rocsolver_add_stream(rocsolver_handle handle, hipStream_t stream) {
//  return rocblas_add_stream(handle, stream);
//}

/*! \deprecated Use \c rocblas_set_stream.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_stream(rocsolver_handle handle, hipStream_t stream)
    __attribute__((deprecated("use rocblas_set_stream")));

/*! \deprecated Use \c rocblas_get_stream.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_stream(rocsolver_handle handle, hipStream_t* stream)
    __attribute__((deprecated("use rocblas_get_stream")));

/*! \deprecated Use \c rocblas_set_vector.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_vector(rocsolver_int n,
                                                       rocsolver_int elem_size,
                                                       const void* x,
                                                       rocsolver_int incx,
                                                       void* y,
                                                       rocsolver_int incy)
    __attribute__((deprecated("use rocblas_set_vector")));

/*! \deprecated Use \c rocblas_get_vector.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_vector(rocsolver_int n,
                                                       rocsolver_int elem_size,
                                                       const void* x,
                                                       rocsolver_int incx,
                                                       void* y,
                                                       rocsolver_int incy)
    __attribute__((deprecated("use rocblas_get_vector")));

/*! \deprecated Use \c rocblas_set_matrix.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_set_matrix(rocsolver_int rows,
                                                       rocsolver_int cols,
                                                       rocsolver_int elem_size,
                                                       const void* a,
                                                       rocsolver_int lda,
                                                       void* b,
                                                       rocsolver_int ldb)
    __attribute__((deprecated("use rocblas_set_matrix")));

/*! \deprecated Use \c rocblas_get_matrix.
 */
ROCSOLVER_EXPORT rocsolver_status rocsolver_get_matrix(rocsolver_int rows,
                                                       rocsolver_int cols,
                                                       rocsolver_int elem_size,
                                                       const void* a,
                                                       rocsolver_int lda,
                                                       void* b,
                                                       rocsolver_int ldb)
    __attribute__((deprecated("use rocblas_get_matrix")));

#pragma GCC diagnostic pop // reenable deprecation warnings

#ifdef __cplusplus
}
#endif

#endif /* ROCSOLVER_ALIASES_H_ */
