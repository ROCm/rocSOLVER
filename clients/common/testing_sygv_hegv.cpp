#include <testing_sygv_hegv.hpp>

template void testing_sygv_hegv<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygv_hegv<false, false, double>(Arguments& argus);
template void testing_sygv_hegv<false, false, float>(Arguments& argus);
template void testing_sygv_hegv<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygv_hegv<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygv_hegv<true, true, double>(Arguments& argus);
template void testing_sygv_hegv<true, true, float>(Arguments& argus);
template void testing_sygv_hegv<false, true, double>(Arguments& argus);
template void testing_sygv_hegv<false, true, float>(Arguments& argus);
