
#include <testing_lacgv.hpp>

#include <client_util.hpp>

#define TESTING_LACGV(...) template void testing_lacgv<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LACGV, FOREACH_COMPLEX_TYPE, APPLY_STAMP)
