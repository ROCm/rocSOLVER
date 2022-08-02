
#include <testing_sytf2_sytrf.hpp>

#include <client_util.hpp>

#define TESTING_SYTF2_SYTRF(...) template void testing_sytf2_sytrf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYTF2_SYTRF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_REAL_TYPE,
            APPLY_STAMP)
