#include <testing_ormtr_unmtr.hpp>

template void testing_ormtr_unmtr<rocblas_double_complex>(Arguments& argus);
template void testing_ormtr_unmtr<rocblas_float_complex>(Arguments& argus);
template void testing_ormtr_unmtr<double>(Arguments& argus);
template void testing_ormtr_unmtr<float>(Arguments& argus);
