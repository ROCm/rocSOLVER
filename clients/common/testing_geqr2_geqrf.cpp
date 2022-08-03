
#include <testing_geqr2_geqrf.hpp>

#define TESTING_GEQR2_GEQRF(...) template void testing_geqr2_geqrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEQR2_GEQRF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
