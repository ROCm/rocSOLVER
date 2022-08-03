
#include <testing_sygv_hegv.hpp>

#define TESTING_SYGV_HEGV(...) template void testing_sygv_hegv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGV_HEGV, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
