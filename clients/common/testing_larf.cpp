
#include <testing_larf.hpp>

#include <client_util.hpp>

#define TESTING_LARF(...) template void testing_larf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LARF, FOREACH_REAL_TYPE, APPLY_STAMP)
