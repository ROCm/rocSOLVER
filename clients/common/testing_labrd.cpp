
#include <testing_labrd.hpp>

#include <client_util.hpp>

#define TESTING_LABRD(...) template void testing_labrd<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LABRD, FOREACH_REAL_TYPE, APPLY_STAMP)
