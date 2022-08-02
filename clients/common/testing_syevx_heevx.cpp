
#include <testing_syevx_heevx.hpp>

#include <client_util.hpp>

#define TESTING_SYEVX_HEEVX(...) template void testing_syevx_heevx<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYEVX_HEEVX, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
