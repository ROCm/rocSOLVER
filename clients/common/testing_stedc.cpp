
#include <testing_stedc.hpp>

#include <client_util.hpp>

#define TESTING_STEDC(...) template void testing_stedc<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_STEDC, FOREACH_REAL_TYPE, APPLY_STAMP)
