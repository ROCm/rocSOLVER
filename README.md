# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of [LAPACK](http://www.netlib.org/lapack/explore-html/index.html) 
functionality on the [ROCm platform](https://rocm.github.io). 

# Documentation

For a detailed description of the rocSOLVER library, its implemented routines, the installation process and user guide, see the
[rocSOLVER documentation](https://rocsolver.readthedocs.io/en/latest).

# Quick start

To download rocSOLVER source code, clone this repository with the command

```bash
git clone https://github.com/ROCmSoftwarePlatform/rocSOLVER.git
```
rocSOLVER requires rocBLAS as a companion GPU BLAS implementation. For more information about rocBLAS and how to
install it, see the [rocBLAS documentation](https://rocblas.readthedocs.io/en/latest).

After a standard installation of rocBLAS, the following commands will build and install rocSOLVER at the standard location
/opt/rocm/rocsolver    

```bash
cd rocsolver 
./install.sh -i
````

Once installed, rocSOLVER can be used just like any other library with a C API. 
The header file will need to be included in the user code, and both the rocBLAS and rocSOLVER shared libraries 
will become link-time and run-time dependencies for the user applciation.

# Using rocSOLVER example

The following code snippet uses rocSOLVER to compute the QR factorization of a general m-by-n real matrix in double precsision. 
For a description of function rocsolver_dgeqrf see the API documentation [here](https://rocsolver.readthedocs.io/en/latest/userguide_api.html#rocsolver-type-geqrf).

```cpp
/////////////////////////////
// example.cpp source code //
/////////////////////////////

#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
#include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver.h> // for all the rocsolver C interfaces and type declarations

int main() {
  rocblas_int M;
  rocblas_int N;
  rocblas_int lda;

  // here is where you would initialize M, N and lda with desired values

  rocsolver_handle handle;
  rocsolver_create_handle(&handle);

  size_t size_A = size_t(lda) * N;          // the size of the array for the matrix
  size_t size_piv = size_t(std::min(M, N)); // the size of array for the Householder scalars

  std::vector<double> hA(size_A);      // creates array for matrix in CPU
  std::vector<double> hIpiv(size_piv); // creates array for householder scalars in CPU

  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);      // allocates memory for matrix in GPU
  hipMalloc(&dIpiv, sizeof(double)*size_piv); // allocates memory for scalars in GPU

  // here is where you would initialize matrix A (array hA) with input data
  // note: matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]

  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);
  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);
  // copy the results back to CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv, so you can use them here

  hipFree(dA);                        // de-allocate GPU memory
  hipFree(dIpiv);
  rocsolver_destroy_handle(handle);   // destroy handle
}
```
Compile command may vary depending on the system and session environment. Here is an example of a common use case

```bash
>> hipcc -I/opt/rocm/include -L/opt/rocm/lib -lrocsolver -lrocblas example.cpp -o example_executable
```
