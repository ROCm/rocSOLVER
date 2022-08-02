
#include <testing_orgxl_ungxl.hpp>

#include <client_util.hpp>

#define TESTING_ORGXL_UNGXL(...) template void testing_orgxl_ungxl<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_ORGXL_UNGXL, FOREACH_SCALAR_TYPE, FOREACH_BOOLEAN_INT, APPLY_STAMP)
