
#include <testing_potf2_potrf.hpp>

#define TESTING_POTF2_POTRF(...) template void testing_potf2_potrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POTF2_POTRF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
