# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of
[LAPACK](https://www.netlib.org/lapack/) functionality on
[AMD ROCm software](https://rocm.docs.amd.com/).

## Documentation

Documentation for rocSOLVER is available at
[https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/).

To build our documentation locally, use the following code:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

To download the rocSOLVER source code, clone this repository with the command:

```bash
git clone https://github.com/ROCmSoftwarePlatform/rocSOLVER.git
```

rocSOLVER requires rocBLAS as a companion GPU BLAS implementation. For more information about
rocBLAS and how to install it, refer to the
[rocBLAS documentation](https://rocm.docs.amd.com/projects/rocBLAS/).

After a standard installation of rocBLAS, the following commands build rocSOLVER and install it to
`/opt/rocm`:

```bash
cd rocSOLVER
./install.sh -i
```

Once installed, rocSOLVER can be used just like any other library with a C-based API. The header file must be included in the user code, and the rocBLAS and rocSOLVER shared libraries become link-time
and run-time dependencies for the user application.

If you're a developer contributing to rocSOLVER, you can run `./scripts/install-hooks` to install the githooks for autoformatting. Before contributing, refer to our
[contributing guidelines](./CONTRIBUTING.md).

## Using rocSOLVER

To compute the QR factorization of a general m-by-n real matrix in double precision using rocSOLVER,
run the following code. You can find a longer version of this example in `example_basic.cpp` (located in
the [samples directory](./clients/samples/). For a description of the `rocsolver_dgeqrf` function, refer to
the [rocSOLVER API documentation](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/lapack.html#rocsolver-type-geqrf).

```cpp
/////////////////////////////
// example.cpp source code //
/////////////////////////////

#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
#include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations

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

The exact command used to compile the example above may vary depending on your system
environment, but this is a typical example:

```bash
/opt/rocm/bin/hipcc -I/opt/rocm/include -c example.cpp
/opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o
```
