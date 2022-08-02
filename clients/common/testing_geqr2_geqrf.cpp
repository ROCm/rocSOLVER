
#include <testing_geqr2_geqrf.hpp>

#include <client_util.hpp>

#define TESTING_GEQR2_GEQRF(...) template void testing_geqr2_geqrf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GEQR2_GEQRF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_REAL_TYPE,
            APPLY_STAMP)
