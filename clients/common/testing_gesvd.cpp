
#include <testing_gesvd.hpp>

#include <client_util.hpp>

#define TESTING_GESVD(...) template void testing_gesvd<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GESVD, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
