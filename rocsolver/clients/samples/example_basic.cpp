#include <algorithm> // for std::min
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for size_t, printf
#include <vector>

// Example: Compute the QR Factorization of a matrix on the GPU

void get_example_matrix(std::vector<double>& hA,
                        rocblas_int& M,
                        rocblas_int& N,
                        rocblas_int& lda) {
  // a *very* small example input; not a very efficient use of the API
  const double A[3][3] = { {  12, -51,   4},
                           {   6, 167, -68},
                           {  -4,  24, -41} };
  M = 3;
  N = 3;
  lda = 3;
  // note: rocsolver matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]
  hA.resize(size_t(lda) * N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      // copy A (2D array) into hA (1D array, column-major)
      hA[i + j*lda] = A[i][j];
    }
  }
}

// We use rocsolver_dgeqrf to factor a real M-by-N matrix, A.
// See https://rocsolver.readthedocs.io/en/latest/userguide_api.html#_CPPv416rocsolver_dgeqrf14rocblas_handleK11rocblas_intK11rocblas_intPdK11rocblas_intPd
// and https://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html
int main() {
  rocblas_int M;          // rows
  rocblas_int N;          // cols
  rocblas_int lda;        // leading dimension
  std::vector<double> hA; // input matrix on CPU
  get_example_matrix(hA, M, N, lda);

  // let's print the input matrix, just to see it
  printf("A = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", hA[i + j*lda]);
    }
    printf(";\n");
  }
  printf("]\n");

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // calculate the sizes of our arrays
  size_t size_A = size_t(lda) * N;          // count of elements in matrix A
  size_t size_piv = size_t(std::min(M, N)); // count of Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  // copy the results back to CPU
  std::vector<double> hIpiv(size_piv); // array for householder scalars on CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // we can print some of the results if we want to see them
  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  // clean up
  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
