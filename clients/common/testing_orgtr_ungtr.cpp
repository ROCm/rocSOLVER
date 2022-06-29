#include <testing_orgtr_ungtr.hpp>

template void testing_orgtr_ungtr<double>(Arguments& argus);
template void testing_orgtr_ungtr<float>(Arguments& argus);
template void testing_orgtr_ungtr<rocblas_double_complex>(Arguments& argus);
template void testing_orgtr_ungtr<rocblas_float_complex>(Arguments& argus);
