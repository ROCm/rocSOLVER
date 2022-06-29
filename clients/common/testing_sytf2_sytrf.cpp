#include <testing_sytf2_sytrf.hpp>

template void testing_sytf2_sytrf<false, true, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 0, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 1, float>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 1, float>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 0, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 1, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 0, float>(Arguments& argus);
