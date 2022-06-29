#include <testing_orgxl_ungxl.hpp>

template void testing_orgxl_ungxl<double, 0>(Arguments& argus);
template void testing_orgxl_ungxl<float, 0>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_float_complex, 0>(Arguments& argus);
template void testing_orgxl_ungxl<double, 1>(Arguments& argus);
template void testing_orgxl_ungxl<float, 1>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_float_complex, 1>(Arguments& argus);
