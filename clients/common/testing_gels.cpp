#include <testing_gels.hpp>

template void testing_gels<false, true, double>(Arguments& argus);
template void testing_gels<false, true, float>(Arguments& argus);
template void testing_gels<true, true, double>(Arguments& argus);
template void testing_gels<true, true, float>(Arguments& argus);
template void testing_gels<false, false, double>(Arguments& argus);
template void testing_gels<false, false, float>(Arguments& argus);
