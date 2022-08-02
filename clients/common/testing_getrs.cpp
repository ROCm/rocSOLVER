
#include <testing_getrs.hpp>

#include <client_util.hpp>

#define TESTING_GETRS(...) template void testing_getrs<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GETRS, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
