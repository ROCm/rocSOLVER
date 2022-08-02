
#include <testing_sygvx_hegvx.hpp>

#include <client_util.hpp>

#define TESTING_SYGVX_HEGVX(...) template void testing_sygvx_hegvx<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYGVX_HEGVX, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
