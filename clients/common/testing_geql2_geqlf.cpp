
#include <testing_geql2_geqlf.hpp>

#include <client_util.hpp>

#define TESTING_GEQL2_GEQLF(...) template void testing_geql2_geqlf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GEQL2_GEQLF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_REAL_TYPE,
            APPLY_STAMP)
