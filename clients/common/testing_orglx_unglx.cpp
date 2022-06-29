#include <testing_orglx_unglx.hpp>

template void testing_orglx_unglx<double, 1>(Arguments& argus);
template void testing_orglx_unglx<float, 1>(Arguments& argus);
template void testing_orglx_unglx<double, 0>(Arguments& argus);
template void testing_orglx_unglx<float, 0>(Arguments& argus);
template void testing_orglx_unglx<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orglx_unglx<rocblas_float_complex, 1>(Arguments& argus);
template void testing_orglx_unglx<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orglx_unglx<rocblas_float_complex, 0>(Arguments& argus);
