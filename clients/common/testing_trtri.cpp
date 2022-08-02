
#include <testing_trtri.hpp>

#include <client_util.hpp>

#define TESTING_TRTRI(...) template void testing_trtri<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_TRTRI, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
