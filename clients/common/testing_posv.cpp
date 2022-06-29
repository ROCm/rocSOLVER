#include <testing_posv.hpp>

template void testing_posv<true, true, double>(Arguments& argus);
template void testing_posv<true, true, float>(Arguments& argus);
template void testing_posv<false, false, double>(Arguments& argus);
template void testing_posv<false, false, float>(Arguments& argus);
template void testing_posv<false, true, double>(Arguments& argus);
template void testing_posv<false, true, float>(Arguments& argus);
