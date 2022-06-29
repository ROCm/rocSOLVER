#include <testing_gelq2_gelqf.hpp>

template void testing_gelq2_gelqf<false, true, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 1, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 0, float>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 1, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 0, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 1, float>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 0, float>(Arguments& argus);
