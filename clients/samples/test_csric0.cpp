#include "rocsparse/rocsparse.h"

#include "assert.h"
#include "stdio.h"

#define CHECK_ROCSPARSE(fcn)                                                   \
  {                                                                            \
    rocsparse_status istat = (fcn);                                            \
    if (istat != rocsparse_status_success) {                                   \
      printf("__FILE__ %s, __LINE__ %d, istat=%d\n", __FILE__, __LINE__,       \
             istat);                                                           \
    };                                                                         \
    assert(istat == rocsparse_status_success);                                 \
  }

#define CHECK_HIP(fcn)                                                         \
  {                                                                            \
    hipError_t istat = (fcn);                                                  \
    assert(istat == hipSuccess);                                               \
  }

int main() {
  // Create rocSPARSE handle
  rocsparse_handle handle;
  CHECK_ROCSPARSE(rocsparse_create_handle(&handle));

  // A = [100 12   13   14]
  //     [12  200  23   24]
  //     [13  23   300  34]
  //     [14  24   34   400]

  int const m = 4;
  int const nnz = m * m;
  double h_csr_val[] = {100, 12, 13,  14, 12, 200, 23, 24,
                        13,  23, 300, 34, 14, 24,  34, 400};

  rocsparse_int h_csr_row_ptr[] = {0, 4, 8, 12, 16};

  rocsparse_int h_csr_col_ind[] = {0, 1, 2, 3, 0, 1, 2, 3,
                                   0, 1, 2, 3, 0, 1, 2, 3};

  double *csr_val = nullptr;
  rocsparse_int *csr_row_ptr = nullptr;
  rocsparse_int *csr_col_ind = nullptr;

  CHECK_HIP(hipMalloc(&csr_val, sizeof(double) * nnz));
  CHECK_HIP(hipMalloc(&csr_row_ptr, sizeof(rocsparse_int) * (m + 1)));
  CHECK_HIP(hipMalloc(&csr_col_ind, sizeof(rocsparse_int) * nnz));

  assert(csr_val != nullptr);
  assert(csr_row_ptr != nullptr);
  assert(csr_col_ind != nullptr);

  CHECK_HIP(hipMemcpy(csr_val, h_csr_val, sizeof(double) * nnz,
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(csr_row_ptr, h_csr_row_ptr,
                      sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(csr_col_ind, h_csr_col_ind, sizeof(rocsparse_int) * (nnz),
                      hipMemcpyHostToDevice));

  double h_x[] = {1.0, 2.0, 3.0, 4.0};
  double h_rhs[] = {0.0, 0.0, 0.0, 0.0};

  int const idebug = 1;

  for (rocsparse_int i = 0; i < m; i++) {
    h_rhs[i] = 0;
    for (rocsparse_int j = h_csr_row_ptr[i]; j < h_csr_row_ptr[i + 1]; j++) {
      rocsparse_int jcol = h_csr_col_ind[j];
      double const xj = h_x[jcol];
      double const aij = h_csr_val[j];

      if (idebug >= 2) {
        printf("i=%d, j=%d, jcol=%d, xj=%le, aij=%le\n", i, j, jcol, xj, aij);
      };

      h_rhs[i] += aij * xj;
    };
  };

  if (idebug >= 1) {
    for (int i = 0; i < m; i++) {
      printf("h_x[%d] = %le, h_rhs[%d] = %le\n", i, h_x[i], i, h_rhs[i]);
    };
  };

  double *x = nullptr;
  double *y = nullptr;
  double *z = nullptr;

  CHECK_HIP(hipMalloc(&x, sizeof(double) * m));
  CHECK_HIP(hipMalloc(&y, sizeof(double) * m));
  CHECK_HIP(hipMalloc(&z, sizeof(double) * m));

  assert(x != nullptr);
  assert(y != nullptr);
  assert(z != nullptr);

  CHECK_HIP(hipMemcpy(x, h_rhs, sizeof(double) * m, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(y, h_rhs, sizeof(double) * m, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(z, h_rhs, sizeof(double) * m, hipMemcpyHostToDevice));

  // Create matrix descriptor for M
  rocsparse_mat_descr descr_M;
  CHECK_ROCSPARSE(rocsparse_create_mat_descr(&descr_M));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_index_base(descr_M, rocsparse_index_base_zero));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_type(descr_M, rocsparse_matrix_type_general));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_storage_mode(descr_M, rocsparse_storage_mode_sorted));

  // Create matrix descriptor for L
  rocsparse_mat_descr descr_L;
  CHECK_ROCSPARSE(rocsparse_create_mat_descr(&descr_L));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_index_base(descr_L, rocsparse_index_base_zero));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_storage_mode(descr_L, rocsparse_storage_mode_sorted));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_fill_mode(descr_L, rocsparse_fill_mode_lower));
  CHECK_ROCSPARSE(
      rocsparse_set_mat_diag_type(descr_L, rocsparse_diag_type_non_unit));

  // Create matrix info structure
  rocsparse_mat_info info;
  CHECK_ROCSPARSE(rocsparse_create_mat_info(&info));

  // Obtain required buffer size
  size_t buffer_size_M;
  size_t buffer_size_L;
  size_t buffer_size_Lt;
  CHECK_ROCSPARSE(rocsparse_dcsric0_buffer_size(
      handle, m, nnz, descr_M, csr_val, csr_row_ptr, csr_col_ind, info,
      &buffer_size_M));

  CHECK_ROCSPARSE(rocsparse_dcsrsv_buffer_size(
      handle, rocsparse_operation_none, m, nnz, descr_L, csr_val, csr_row_ptr,
      csr_col_ind, info, &buffer_size_L));

  CHECK_ROCSPARSE(rocsparse_dcsrsv_buffer_size(
      handle, rocsparse_operation_transpose, m, nnz, descr_L, csr_val,
      csr_row_ptr, csr_col_ind, info, &buffer_size_Lt));

  size_t buffer_size =
      max(buffer_size_M, max(buffer_size_L, buffer_size_Lt)) + 1;

  // Allocate temporary buffer
  void *temp_buffer = nullptr;
  CHECK_HIP(hipMalloc(&temp_buffer, buffer_size));
  assert(temp_buffer != nullptr);

  // Perform analysis steps, using rocsparse_analysis_policy_reuse to improve
  // computation performance
  CHECK_ROCSPARSE(rocsparse_dcsric0_analysis(
      handle, m, nnz, descr_M, csr_val, csr_row_ptr, csr_col_ind, info,
      rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto,
      temp_buffer));

  CHECK_ROCSPARSE(rocsparse_dcsrsv_analysis(
      handle, rocsparse_operation_none, m, nnz, descr_L, csr_val, csr_row_ptr,
      csr_col_ind, info, rocsparse_analysis_policy_reuse,
      rocsparse_solve_policy_auto, temp_buffer));

  CHECK_ROCSPARSE(rocsparse_dcsrsv_analysis(
      handle, rocsparse_operation_transpose, m, nnz, descr_L, csr_val,
      csr_row_ptr, csr_col_ind, info, rocsparse_analysis_policy_reuse,
      rocsparse_solve_policy_auto, temp_buffer));

  // Check for zero pivot
  rocsparse_int position;
  if (rocsparse_status_zero_pivot ==
      rocsparse_csric0_zero_pivot(handle, info, &position)) {
    printf("A has structural zero at A(%d,%d)\n", position, position);
  }

  // Compute incomplete Cholesky factorization M = LL'
  CHECK_ROCSPARSE(rocsparse_dcsric0(handle, m, nnz, descr_M, csr_val,
                                    csr_row_ptr, csr_col_ind, info,
                                    rocsparse_solve_policy_auto, temp_buffer));

  // Check for zero pivot
  if (rocsparse_status_zero_pivot ==
      rocsparse_csric0_zero_pivot(handle, info, &position)) {
    printf("L has structural and/or numerical zero at L(%d,%d)\n", position,
           position);
  }

  double alpha = 1.0;
  // Solve Lz = x
  CHECK_ROCSPARSE(
      rocsparse_dcsrsv_solve(handle, rocsparse_operation_none, m, nnz, &alpha,
                             descr_L, csr_val, csr_row_ptr, csr_col_ind, info,
                             x, z, rocsparse_solve_policy_auto, temp_buffer));

  // Solve L'y = z
  CHECK_ROCSPARSE(rocsparse_dcsrsv_solve(
      handle, rocsparse_operation_transpose, m, nnz, &alpha, descr_L, csr_val,
      csr_row_ptr, csr_col_ind, info, z, y, rocsparse_solve_policy_auto,
      temp_buffer));

  double h_y[] = {0, 0, 0, 0};
  double h_z[] = {0, 0, 0, 0};
  CHECK_HIP(hipMemcpy(h_y, y, sizeof(double) * m, hipMemcpyDeviceToHost));
  CHECK_HIP(hipMemcpy(h_z, z, sizeof(double) * m, hipMemcpyDeviceToHost));

  if (idebug >= 1) {
    for (int i = 0; i < m; i++) {
      printf("h_y[%d] = %le, h_z[%d] = %le\n", i, h_y[i], i, h_z[i]);
    };
  };

  if (idebug >= 2) {
    CHECK_HIP(hipMemcpy(h_csr_val, csr_val, sizeof(double) * nnz,
                        hipMemcpyDeviceToHost));
    for (int i = 0; i < m; i++) {
      for (int j = h_csr_row_ptr[i]; j < h_csr_row_ptr[i + 1]; j++) {
        int jcol = h_csr_col_ind[j];
        double Mij = h_csr_val[j];

        printf("M(%d,%d) = %le\n", i, jcol, Mij);
      };
    };
  };

  // Clean up
  CHECK_HIP(hipFree(temp_buffer));
  CHECK_ROCSPARSE(rocsparse_destroy_mat_info(info));
  CHECK_ROCSPARSE(rocsparse_destroy_mat_descr(descr_M));
  CHECK_ROCSPARSE(rocsparse_destroy_mat_descr(descr_L));
  CHECK_ROCSPARSE(rocsparse_destroy_handle(handle));
}
