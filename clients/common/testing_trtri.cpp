#include <testing_trtri.hpp>

template void testing_trtri<true, true, double>(Arguments& argus);
template void testing_trtri<true, true, float>(Arguments& argus);
template void testing_trtri<false, false, double>(Arguments& argus);
template void testing_trtri<false, false, float>(Arguments& argus);
template void testing_trtri<false, true, double>(Arguments& argus);
template void testing_trtri<false, true, float>(Arguments& argus);
