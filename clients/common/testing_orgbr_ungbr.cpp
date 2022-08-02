
#include <testing_orgbr_ungbr.hpp>

#include <client_util.hpp>

#define TESTING_ORGBR_UNGBR(...) template void testing_orgbr_ungbr<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_ORGBR_UNGBR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
