
#include <testing_gesv.hpp>

#include <client_util.hpp>

#define TESTING_GESV(...) template void testing_gesv<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GESV, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
