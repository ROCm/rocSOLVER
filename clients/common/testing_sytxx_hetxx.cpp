
#include <testing_sytxx_hetxx.hpp>

#include <client_util.hpp>

#define TESTING_SYTXX_HETXX(...) template void testing_sytxx_hetxx<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYTXX_HETXX,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
