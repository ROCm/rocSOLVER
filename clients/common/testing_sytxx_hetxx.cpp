#include <testing_sytxx_hetxx.hpp>

template void testing_sytxx_hetxx<false, false, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, float>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, float>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, float>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, float>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, float>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, float>(Arguments& argus);
