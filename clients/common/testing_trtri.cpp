
#include <testing_trtri.hpp>

#define TESTING_TRTRI(...) template void testing_trtri<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_TRTRI, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
