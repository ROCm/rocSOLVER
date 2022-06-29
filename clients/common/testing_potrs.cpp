#include <testing_potrs.hpp>

template void testing_potrs<true, true, double>(Arguments& argus);
template void testing_potrs<true, true, float>(Arguments& argus);
template void testing_potrs<false, true, double>(Arguments& argus);
template void testing_potrs<false, true, float>(Arguments& argus);
template void testing_potrs<false, false, double>(Arguments& argus);
template void testing_potrs<false, false, float>(Arguments& argus);
