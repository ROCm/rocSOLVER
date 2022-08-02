
#include <testing_syev_heev.hpp>

#include <client_util.hpp>

#define TESTING_SYEV_HEEV(...) template void testing_syev_heev<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYEV_HEEV, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
