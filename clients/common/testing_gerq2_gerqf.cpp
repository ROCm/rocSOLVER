#include <testing_gerq2_gerqf.hpp>

template void testing_gerq2_gerqf<true, true, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 0, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 1, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 0, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 1, float>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 1, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 0, float>(Arguments& argus);
