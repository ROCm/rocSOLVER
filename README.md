# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of [LAPACK][1]
functionality on the [ROCm platform][2].

## Documentation

For a detailed description of the rocSOLVER library, its implemented routines,
the installation process and user guide, see the [rocSOLVER documentation][3].

## Building rocSOLVER

To download the rocSOLVER source code, clone this repository with the command:

    git clone https://github.com/ROCmSoftwarePlatform/rocSOLVER.git

rocSOLVER requires rocBLAS as a companion GPU BLAS implementation. For
more information about rocBLAS and how to install it, see the
[rocBLAS documentation][4].

After a standard installation of rocBLAS, the following commands will build
rocSOLVER and install to `/opt/rocm/rocsolver`:

    cd rocSOLVER
    ./install.sh -i

Once installed, rocSOLVER can be used just like any other library with a C API.
The header file will need to be included in the user code, and both the rocBLAS
and rocSOLVER shared libraries will become link-time and run-time dependencies
for the user application.

If you are a developer contributing to rocSOLVER, you may wish to run
`.githooks/install` to install the git hooks for autoformatting.

## Using rocSOLVER

The following code snippet shows how to compute the QR factorization of a
general m-by-n real matrix in double precision using rocSOLVER. A longer
version of this example is provided in the [rocSOLVER samples][5].
For a description of the function `rocsolver_dgeqrf`, see the
[API documentation][6].

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

  rocblas_handle handle;
  rocblas_create_handle(&handle);

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
  rocblas_destroy_handle(handle);     // destroy handle
}
```

The exact command used to compile the example above may vary depending on the
system environment, but here is a typical example:

    /opt/rocm/bin/hipcc -I/opt/rocm/include -c example.cpp
    /opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o


[1]: https://www.netlib.org/lapack/explore-html/index.html
[2]: https://rocm.github.io
[3]: https://rocsolver.readthedocs.io/en/latest
[4]: https://rocblas.readthedocs.io/en/latest
[5]: rocsolver/clients/samples/example_basic.cpp
[6]: https://rocsolver.readthedocs.io/en/latest/userguide_api.html#rocsolver-type-geqrf
