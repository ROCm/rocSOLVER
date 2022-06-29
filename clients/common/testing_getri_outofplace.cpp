#include <testing_getri_outofplace.hpp>

template void testing_getri_outofplace<false, true, double>(Arguments& argus);
template void testing_getri_outofplace<false, true, float>(Arguments& argus);
template void testing_getri_outofplace<false, false, double>(Arguments& argus);
template void testing_getri_outofplace<false, false, float>(Arguments& argus);
template void testing_getri_outofplace<true, true, double>(Arguments& argus);
template void testing_getri_outofplace<true, true, float>(Arguments& argus);
