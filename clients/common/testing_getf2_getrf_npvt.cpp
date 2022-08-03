
#include <testing_getf2_getrf_npvt.hpp>

#define TESTING_GETF2_GETRF_NPVT(...) \
    template void testing_getf2_getrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETF2_GETRF_NPVT,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
