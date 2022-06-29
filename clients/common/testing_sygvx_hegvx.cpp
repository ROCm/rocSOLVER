#include <testing_sygvx_hegvx.hpp>

template void testing_sygvx_hegvx<true, true, double>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, float>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, double>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, float>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, double>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, float>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, rocblas_float_complex>(Arguments& argus);
