#include <testing_potf2_potrf.hpp>

template void testing_potf2_potrf<false, false, 0, double>(Arguments& argus);
template void testing_potf2_potrf<false, false, 0, float>(Arguments& argus);
template void testing_potf2_potrf<false, true, 0, double>(Arguments& argus);
template void testing_potf2_potrf<false, true, 0, float>(Arguments& argus);
template void testing_potf2_potrf<false, true, 1, double>(Arguments& argus);
template void testing_potf2_potrf<false, true, 1, float>(Arguments& argus);
template void testing_potf2_potrf<false, false, 1, double>(Arguments& argus);
template void testing_potf2_potrf<false, false, 1, float>(Arguments& argus);
template void testing_potf2_potrf<true, true, 1, double>(Arguments& argus);
template void testing_potf2_potrf<true, true, 1, float>(Arguments& argus);
template void testing_potf2_potrf<true, true, 0, double>(Arguments& argus);
template void testing_potf2_potrf<true, true, 0, float>(Arguments& argus);
