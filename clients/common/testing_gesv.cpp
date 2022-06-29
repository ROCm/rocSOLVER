#include <testing_gesv.hpp>

template void testing_gesv<false, true, double>(Arguments& argus);
template void testing_gesv<false, true, float>(Arguments& argus);
template void testing_gesv<false, false, double>(Arguments& argus);
template void testing_gesv<false, false, float>(Arguments& argus);
template void testing_gesv<true, true, double>(Arguments& argus);
template void testing_gesv<true, true, float>(Arguments& argus);
