
#include <testing_latrd.hpp>

#include <client_util.hpp>

#define TESTING_LATRD(...) template void testing_latrd<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LATRD, FOREACH_REAL_TYPE, APPLY_STAMP)
