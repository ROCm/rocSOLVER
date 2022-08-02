
#include <testing_potrs.hpp>

#include <client_util.hpp>

#define TESTING_POTRS(...) template void testing_potrs<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_POTRS, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
