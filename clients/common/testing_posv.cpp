
#include <testing_posv.hpp>

#define TESTING_POSV(...) template void testing_posv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POSV, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
