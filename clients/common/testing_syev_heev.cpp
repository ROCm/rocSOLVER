#include <testing_syev_heev.hpp>

template void testing_syev_heev<false, true, double>(Arguments& argus);
template void testing_syev_heev<false, true, float>(Arguments& argus);
template void testing_syev_heev<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syev_heev<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_syev_heev<true, true, double>(Arguments& argus);
template void testing_syev_heev<true, true, float>(Arguments& argus);
template void testing_syev_heev<false, false, double>(Arguments& argus);
template void testing_syev_heev<false, false, float>(Arguments& argus);
template void testing_syev_heev<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<false, true, rocblas_float_complex>(Arguments& argus);
