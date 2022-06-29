#include <testing_orgbr_ungbr.hpp>

template void testing_orgbr_ungbr<double>(Arguments& argus);
template void testing_orgbr_ungbr<float>(Arguments& argus);
template void testing_orgbr_ungbr<rocblas_double_complex>(Arguments& argus);
template void testing_orgbr_ungbr<rocblas_float_complex>(Arguments& argus);
