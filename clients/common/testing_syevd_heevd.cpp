#include <testing_syevd_heevd.hpp>

template void testing_syevd_heevd<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevd_heevd<false, true, double>(Arguments& argus);
template void testing_syevd_heevd<false, true, float>(Arguments& argus);
template void testing_syevd_heevd<true, true, double>(Arguments& argus);
template void testing_syevd_heevd<true, true, float>(Arguments& argus);
template void testing_syevd_heevd<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_syevd_heevd<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevd_heevd<false, false, double>(Arguments& argus);
template void testing_syevd_heevd<false, false, float>(Arguments& argus);
