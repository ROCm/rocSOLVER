
#include <testing_syevj_heevj.hpp>

#define TESTING_SYEVJ_HEEVJ(...) template void testing_syevj_heevj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVJ_HEEVJ, FOREACH_BOOLEAN_0, FOREACH_BOOLEAN_1, FOREACH_SCALAR_TYPE, APPLY_STAMP)
