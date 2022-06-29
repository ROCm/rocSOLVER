#include <testing_getri.hpp>

template void testing_getri<false, true, double>(Arguments& argus);
template void testing_getri<false, true, float>(Arguments& argus);
template void testing_getri<false, false, double>(Arguments& argus);
template void testing_getri<false, false, float>(Arguments& argus);
template void testing_getri<true, true, double>(Arguments& argus);
template void testing_getri<true, true, float>(Arguments& argus);
