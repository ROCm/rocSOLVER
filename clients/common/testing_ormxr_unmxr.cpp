#include <testing_ormxr_unmxr.hpp>

template void testing_ormxr_unmxr<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_float_complex, 0>(Arguments& argus);
template void testing_ormxr_unmxr<double, 1>(Arguments& argus);
template void testing_ormxr_unmxr<float, 1>(Arguments& argus);
template void testing_ormxr_unmxr<double, 0>(Arguments& argus);
template void testing_ormxr_unmxr<float, 0>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_float_complex, 1>(Arguments& argus);
