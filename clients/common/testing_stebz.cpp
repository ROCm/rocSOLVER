
#include <testing_stebz.hpp>

#include <client_util.hpp>

#define TESTING_STEBZ(...) template void testing_stebz<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_STEBZ, FOREACH_REAL_TYPE, APPLY_STAMP)
