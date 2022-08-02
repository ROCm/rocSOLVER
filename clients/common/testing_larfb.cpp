
#include <testing_larfb.hpp>

#include <client_util.hpp>

#define TESTING_LARFB(...) template void testing_larfb<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LARFB, FOREACH_REAL_TYPE, APPLY_STAMP)
