#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc

#define HIP_CHECK(...)                                         \
    {                                                          \
        hipError_t _status = (__VA_ARGS__);                    \
        if(_status != hipSuccess)       {                       \
            printf("hipError: %d\n", _status); \
            return _status;} \
    }

int main(int argc, char** argv) {
    int n;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    else {
        n = 8;
    }

    printf("graph capturing %d loops\n", n);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    hipStream_t exec_stream;
    hipStream_t capture_stream;

    HIP_CHECK(hipStreamCreate(&exec_stream));
    HIP_CHECK(hipStreamCreate(&capture_stream));

    rocblas_set_stream(handle, capture_stream);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    HIP_CHECK(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeGlobal));

    double* x;
    HIP_CHECK(hipMallocAsync(&x, sizeof(double) * n, capture_stream));
    HIP_CHECK(hipMemsetAsync(x, 0, n * sizeof(double), capture_stream));
    double alpha = 1, tau = 1;

    for (int i = 0; i < n; i++) {
        rocsolver_dlarfg(handle, n, &alpha, x, 1, &tau);
    }

    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(capture_stream, &graph));
    printf("graph capture complete\n");
    rocblas_set_stream(handle, exec_stream);

    hipGraphExec_t exec;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, NULL, NULL, 0));
    printf("launching graph...");
    HIP_CHECK(hipGraphLaunch(exec, exec_stream));
    printf("OK!\n");

    printf("synchronizing stream...");
    HIP_CHECK(hipStreamSynchronize(exec_stream));
    printf("OK!\n");

    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipStreamDestroy(capture_stream));
    HIP_CHECK(hipStreamDestroy(exec_stream));
}
