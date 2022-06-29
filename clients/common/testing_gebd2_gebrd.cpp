#include <testing_gebd2_gebrd.hpp>

template void testing_gebd2_gebrd<false, true, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 1, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 0, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 1, float>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 0, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 0, float>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 1, float>(Arguments& argus);
