#include <testing_geql2_geqlf.hpp>

template void testing_geql2_geqlf<false, true, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 1, float>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 1, float>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 0, float>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 0, float>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 0, float>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 1, float>(Arguments& argus);
