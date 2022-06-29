#include <testing_getri_npvt.hpp>

template void testing_getri_npvt<false, true, double>(Arguments& argus);
template void testing_getri_npvt<false, true, float>(Arguments& argus);
template void testing_getri_npvt<true, true, double>(Arguments& argus);
template void testing_getri_npvt<true, true, float>(Arguments& argus);
template void testing_getri_npvt<false, false, double>(Arguments& argus);
template void testing_getri_npvt<false, false, float>(Arguments& argus);
