
#include <testing_larft.hpp>

#include <client_util.hpp>

#define TESTING_LARFT(...) template void testing_larft<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_LARFT, FOREACH_REAL_TYPE, APPLY_STAMP)
