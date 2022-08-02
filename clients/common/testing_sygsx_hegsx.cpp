
#include <testing_sygsx_hegsx.hpp>

#include <client_util.hpp>

#define TESTING_SYGSX_HEGSX(...) template void testing_sygsx_hegsx<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYGSX_HEGSX,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
