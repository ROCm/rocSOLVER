
#include <testing_gelq2_gelqf.hpp>

#include <client_util.hpp>

#define TESTING_GELQ2_GELQF(...) template void testing_gelq2_gelqf<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_GELQ2_GELQF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_REAL_TYPE,
            APPLY_STAMP)
