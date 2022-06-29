#include <testing_potri.hpp>

template void testing_potri<false, true, double>(Arguments& argus);
template void testing_potri<false, true, float>(Arguments& argus);
template void testing_potri<true, true, double>(Arguments& argus);
template void testing_potri<true, true, float>(Arguments& argus);
template void testing_potri<false, false, double>(Arguments& argus);
template void testing_potri<false, false, float>(Arguments& argus);
