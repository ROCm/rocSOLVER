#include <testing_sygvj_hegvj.hpp>

template void testing_sygvj_hegvj<false, true, double>(Arguments& argus);
template void testing_sygvj_hegvj<false, true, float>(Arguments& argus);
template void testing_sygvj_hegvj<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvj_hegvj<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvj_hegvj<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvj_hegvj<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvj_hegvj<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygvj_hegvj<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygvj_hegvj<true, true, double>(Arguments& argus);
template void testing_sygvj_hegvj<true, true, float>(Arguments& argus);
template void testing_sygvj_hegvj<false, false, double>(Arguments& argus);
template void testing_sygvj_hegvj<false, false, float>(Arguments& argus);
