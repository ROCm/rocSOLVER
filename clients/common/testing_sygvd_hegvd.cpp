#include <testing_sygvd_hegvd.hpp>

template void testing_sygvd_hegvd<false, false, double>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, float>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, double>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, float>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, double>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, float>(Arguments& argus);
