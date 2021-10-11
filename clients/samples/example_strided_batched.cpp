#include <algorithm> // for std::min
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for size_t, printf
#include <vector>

// Example: Compute the QR Factorizations of an array of matrices on the GPU

void get_example_matrices(std::vector<double>& hA,
                        rocblas_int& M,
                        rocblas_int& N,
                        rocblas_int& lda,
                        rocblas_stride& strideA,
                        rocblas_int& batch_count) {
  const double A[2][3][3] = {
    // First input matrix
    { { 12, -51,   4},
      {  6, 167, -68},
      { -4,  24, -41} },

    // Second input matrix
    { {  3, -12,  11},
      {  4, -46,  -2},
      {  0,   5,  15} } };

  M = 3;
  N = 3;
  lda = 3;
  batch_count = 2;
  strideA = lda * N;
  hA.resize(strideA * batch_count);
  // copy A (3D array) into hA (1D array, column-major)
  for (size_t b = 0; b < batch_count; ++b)
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        hA[i + j*lda + b*strideA] = A[b][i][j];
}

// Use rocsolver_dgeqrf_strided_batched to factor an array of real M-by-N matrices.
int main() {
  rocblas_int M;           // rows
  rocblas_int N;           // cols
  rocblas_int lda;         // leading dimension
  rocblas_stride strideA;  // stride from start of one matrix to the next
  rocblas_int batch_count; // number of matricies
  std::vector<double> hA;  // input matrix data on CPU
  get_example_matrices(hA, M, N, lda, strideA, batch_count);

  // print the input matrices
  for (size_t b = 0; b < batch_count; ++b) {
    printf("A[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", hA[i + j*lda + strideA*b]);
      }
      printf(";\n");
    }
    printf("]\n");
  }

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // calculate the sizes of our arrays
  size_t size_A = size_t(strideA) * batch_count;   // elements in array for matrices
  rocblas_stride strideP = std::min(M, N);         // stride of Householder scalar sets
  size_t size_piv = size_t(strideP) * batch_count; // elements in array for Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  // compute the QR factorizations on the GPU
  rocsolver_dgeqrf_strided_batched(handle, M, N, dA, lda, strideA, dIpiv, strideP, batch_count);

  // copy the results back to CPU
  std::vector<double> hIpiv(size_piv); // array for householder scalars on CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // print some of the results
  for (size_t b = 0; b < batch_count; ++b) {
    printf("R[%zu] = [\n", b);
    for (size_t i = 0; i < M; ++i) {
      printf("  ");
      for (size_t j = 0; j < N; ++j) {
        printf("% 4.f ", (i <= j) ? hA[i + j*lda] : 0);
      }
      printf(";\n");
    }
    printf("]\n");
  }

  // clean up
  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
