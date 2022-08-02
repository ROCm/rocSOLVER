
#include <testing_lasyf.hpp>

#include <client_util.hpp>

#define TESTING_LASYF(...) template void testing_lasyf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LASYF, FOREACH_REAL_TYPE, APPLY_STAMP)
