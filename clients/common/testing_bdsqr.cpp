
#include <testing_bdsqr.hpp>

#include <client_util.hpp>

#define TESTING_BDSQR(...) template void testing_bdsqr<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_BDSQR, FOREACH_REAL_TYPE, APPLY_STAMP)
