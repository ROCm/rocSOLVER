#include <testing_geqr2_geqrf.hpp>

template void testing_geqr2_geqrf<false, false, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 0, float>(Arguments& argus);
template void testing_geqr2_geqrf<true, false, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, false, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 0, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 0, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 1, float>(Arguments& argus);
