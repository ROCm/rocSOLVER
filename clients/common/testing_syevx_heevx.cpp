#include <testing_syevx_heevx.hpp>

template void testing_syevx_heevx<false, false, double>(Arguments& argus);
template void testing_syevx_heevx<false, false, float>(Arguments& argus);
template void testing_syevx_heevx<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevx_heevx<false, true, double>(Arguments& argus);
template void testing_syevx_heevx<false, true, float>(Arguments& argus);
template void testing_syevx_heevx<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevx_heevx<true, true, double>(Arguments& argus);
template void testing_syevx_heevx<true, true, float>(Arguments& argus);
template void testing_syevx_heevx<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<false, false, rocblas_float_complex>(Arguments& argus);
