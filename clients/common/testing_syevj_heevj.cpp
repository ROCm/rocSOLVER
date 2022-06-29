#include <testing_syevj_heevj.hpp>

template void testing_syevj_heevj<false, false, double>(Arguments& argus);
template void testing_syevj_heevj<false, false, float>(Arguments& argus);
template void testing_syevj_heevj<false, true, double>(Arguments& argus);
template void testing_syevj_heevj<false, true, float>(Arguments& argus);
template void testing_syevj_heevj<true, true, double>(Arguments& argus);
template void testing_syevj_heevj<true, true, float>(Arguments& argus);
template void testing_syevj_heevj<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syevj_heevj<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_syevj_heevj<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevj_heevj<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevj_heevj<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevj_heevj<false, true, rocblas_float_complex>(Arguments& argus);
