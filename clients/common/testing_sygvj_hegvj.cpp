
#include <testing_sygvj_hegvj.hpp>

#include <client_util.hpp>

#define TESTING_SYGVJ_HEGVJ(...) template void testing_sygvj_hegvj<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_SYGVJ_HEGVJ, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
