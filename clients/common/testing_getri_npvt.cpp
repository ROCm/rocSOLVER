
#include <testing_getri_npvt.hpp>

#include <client_util.hpp>

#define TESTING_GETRI_NPVT(...) template void testing_getri_npvt<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GETRI_NPVT, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_REAL_TYPE, APPLY_STAMP)
