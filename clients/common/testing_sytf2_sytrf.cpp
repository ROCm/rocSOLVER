
#include <testing_sytf2_sytrf.hpp>

#define TESTING_SYTF2_SYTRF(...) template void testing_sytf2_sytrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYTF2_SYTRF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
