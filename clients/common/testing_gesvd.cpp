
#include <testing_gesvd.hpp>

#define TESTING_GESVD(...) template void testing_gesvd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GESVD, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
