
#include <testing_getri_outofplace.hpp>

#define TESTING_GETRI_OUTOFPLACE(...) \
    template void testing_getri_outofplace<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRI_OUTOFPLACE, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
