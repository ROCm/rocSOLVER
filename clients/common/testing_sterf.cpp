
#include <testing_sterf.hpp>

#include <client_util.hpp>

#define TESTING_STERF(...) template void testing_sterf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_STERF, FOREACH_REAL_TYPE, APPLY_STAMP)
