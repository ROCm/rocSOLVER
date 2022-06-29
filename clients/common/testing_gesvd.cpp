#include <testing_gesvd.hpp>

template void testing_gesvd<false, false, double>(Arguments& argus);
template void testing_gesvd<false, false, float>(Arguments& argus);
template void testing_gesvd<true, true, double>(Arguments& argus);
template void testing_gesvd<true, true, float>(Arguments& argus);
template void testing_gesvd<false, true, double>(Arguments& argus);
template void testing_gesvd<false, true, float>(Arguments& argus);
