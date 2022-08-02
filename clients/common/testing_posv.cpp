
#include <testing_posv.hpp>

#include <client_util.hpp>

#define TESTING_POSV(...) template void testing_posv<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_POSV, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
