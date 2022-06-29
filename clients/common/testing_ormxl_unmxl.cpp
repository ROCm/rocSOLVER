#include <testing_ormxl_unmxl.hpp>

template void testing_ormxl_unmxl<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_float_complex, 0>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_float_complex, 1>(Arguments& argus);
template void testing_ormxl_unmxl<double, 0>(Arguments& argus);
template void testing_ormxl_unmxl<float, 0>(Arguments& argus);
template void testing_ormxl_unmxl<double, 1>(Arguments& argus);
template void testing_ormxl_unmxl<float, 1>(Arguments& argus);
