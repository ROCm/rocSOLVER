
#include <testing_laswp.hpp>

#include <client_util.hpp>

#define TESTING_LASWP(...) template void testing_laswp<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LASWP, FOREACH_REAL_TYPE, APPLY_STAMP)
