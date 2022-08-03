
#include <testing_getri.hpp>

#define TESTING_GETRI(...) template void testing_getri<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRI, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
