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
For a description of function rocsolver_dgeqrf see the API documentation [here](https://rocsolver.readthedocs.io/en/latest/api.html#rocsolver-type-geqrf).

```C
///////////////////////////
// example.c source code //
///////////////////////////

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <rocsolver.h>      // this includes the rocsolver header

using namespace std;

int main() {
    rocsolver_int M;
    rocsolver_int N;
    rocsolver_int lda;

    // initialize M, N and lda with desired values
    // here===>>

    rocsolver_handle handle;
    rocsolver_create_handle(&handle); // this creates the rocsolver handle

    rocsolver_int size_A = lda * N;     // this is the size of the array that will hold the matrix
    rocsolver_int size_piv = min(M, N); // this is size of array that will have the Householder scalars   

    vector<double> hA(size_A);        // creates array for matrix in CPU
    vector<double> hIpiv(size_piv);   // creates array for householder scalars in CPU

    double *dA, *dIpiv;
    hipMalloc(&dA,sizeof(double)*size_A);       // allocates memory for matrix in GPU
    hipMalloc(&dIpiv,sizeof(double)*size_piv);  // allocates memory for scalars in GPU
  
    // initialize matrix A (array hA) with input data
    // here===>>

    hipMemcpy(dA,hA.data(),sizeof(double)*size_A,hipMemcpyHostToDevice); // copy data to GPU
    rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);                      // compute the QR factorization on the GPU   
    hipMemcpy(hA.data(),dA,sizeof(double)*size_A,hipMemcpyDeviceToHost); // copy the results back to CPU
    hipMemcpy(hIpiv.data(),dIpiv,sizeof(double)*size_piv,hipMemcpyDeviceToHost);

    // do something with the results in hA and hIpiv
    // here===>>

    hipFree(dA);                        // de-allocate GPU memory 
    hipFree(dIpiv);
    rocsolver_destroy_handle(handle);   // destroy handle
  
    return 0;
}
```
Compile command may vary depending on the system and session environment. Here is an example of a common use case

```bash
>> hipcc -I/opt/rocm/include -L/opt/rocm/lib -lrocsolver -lrocblas example.c -o example.exe            
```

