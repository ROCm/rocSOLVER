#include <testing_getrs.hpp>

template void testing_getrs<false, true, double>(Arguments& argus);
template void testing_getrs<false, true, float>(Arguments& argus);
template void testing_getrs<false, false, double>(Arguments& argus);
template void testing_getrs<false, false, float>(Arguments& argus);
template void testing_getrs<true, true, double>(Arguments& argus);
template void testing_getrs<true, true, float>(Arguments& argus);
