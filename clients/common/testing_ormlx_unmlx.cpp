#include <testing_ormlx_unmlx.hpp>

template void testing_ormlx_unmlx<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_float_complex, 1>(Arguments& argus);
template void testing_ormlx_unmlx<double, 0>(Arguments& argus);
template void testing_ormlx_unmlx<float, 0>(Arguments& argus);
template void testing_ormlx_unmlx<double, 1>(Arguments& argus);
template void testing_ormlx_unmlx<float, 1>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_float_complex, 0>(Arguments& argus);
