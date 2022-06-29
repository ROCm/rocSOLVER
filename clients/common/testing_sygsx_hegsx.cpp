#include <testing_sygsx_hegsx.hpp>

template void testing_sygsx_hegsx<false, false, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, float>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, float>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, rocblas_float_complex>(Arguments& argus);
