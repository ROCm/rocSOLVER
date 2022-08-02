
#include <testing_steqr.hpp>

#include <client_util.hpp>

#define TESTING_STEQR(...) template void testing_steqr<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_STEQR, FOREACH_REAL_TYPE, APPLY_STAMP)
