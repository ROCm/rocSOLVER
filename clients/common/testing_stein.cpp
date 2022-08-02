
#include <testing_stein.hpp>

#include <client_util.hpp>

#define TESTING_STEIN(...) template void testing_stein<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_STEIN, FOREACH_REAL_TYPE, APPLY_STAMP)
