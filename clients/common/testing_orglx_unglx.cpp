
#include <testing_orglx_unglx.hpp>

#include <client_util.hpp>

#define TESTING_ORGLX_UNGLX(...) template void testing_orglx_unglx<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_ORGLX_UNGLX, FOREACH_SCALAR_TYPE, FOREACH_BOOLEAN_INT, APPLY_STAMP)
