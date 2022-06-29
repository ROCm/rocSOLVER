#include <testing_getf2_getrf.hpp>

template void testing_getf2_getrf<true, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf<true, true, 1, float>(Arguments& argus);
template void testing_getf2_getrf<false, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf<false, true, 1, float>(Arguments& argus);
template void testing_getf2_getrf<false, false, 1, double>(Arguments& argus);
template void testing_getf2_getrf<false, false, 1, float>(Arguments& argus);
template void testing_getf2_getrf<false, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf<false, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf<true, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf<true, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf<false, false, 0, double>(Arguments& argus);
template void testing_getf2_getrf<false, false, 0, float>(Arguments& argus);
