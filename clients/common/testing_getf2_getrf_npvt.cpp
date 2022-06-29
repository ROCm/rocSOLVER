#include <testing_getf2_getrf_npvt.hpp>

template void testing_getf2_getrf_npvt<false, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 1, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 1, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 1, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 0, float>(Arguments& argus);
