
#include <testing_gels.hpp>

#include <client_util.hpp>

#define TESTING_GELS(...) template void testing_gels<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GELS, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
