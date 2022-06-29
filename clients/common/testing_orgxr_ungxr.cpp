#include <testing_orgxr_ungxr.hpp>

template void testing_orgxr_ungxr<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_float_complex, 1>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_float_complex, 0>(Arguments& argus);
template void testing_orgxr_ungxr<double, 1>(Arguments& argus);
template void testing_orgxr_ungxr<float, 1>(Arguments& argus);
template void testing_orgxr_ungxr<double, 0>(Arguments& argus);
template void testing_orgxr_ungxr<float, 0>(Arguments& argus);
