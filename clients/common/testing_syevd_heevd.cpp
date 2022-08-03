
#include <testing_syevd_heevd.hpp>

#define TESTING_SYEVD_HEEVD(...) template void testing_syevd_heevd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVD_HEEVD, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
