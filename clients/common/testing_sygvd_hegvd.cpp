
#include <testing_sygvd_hegvd.hpp>

#include <client_util.hpp>

#define TESTING_SYGVD_HEGVD(...) template void testing_sygvd_hegvd<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYGVD_HEGVD, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
