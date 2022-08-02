
#include <testing_larfg.hpp>

#include <client_util.hpp>

#define TESTING_LARFG(...) template void testing_larfg<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LARFG, FOREACH_REAL_TYPE, APPLY_STAMP)
