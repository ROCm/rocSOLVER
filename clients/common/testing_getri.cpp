
#include <testing_getri.hpp>

#include <client_util.hpp>

#define TESTING_GETRI(...) template void testing_getri<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GETRI, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
