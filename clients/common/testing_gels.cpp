
#include <testing_gels.hpp>

#define TESTING_GELS(...) template void testing_gels<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GELS, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
