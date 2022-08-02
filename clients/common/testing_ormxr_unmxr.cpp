
#include <testing_ormxr_unmxr.hpp>

#include <client_util.hpp>

#define TESTING_ORMXR_UNMXR(...) template void testing_ormxr_unmxr<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_ORMXR_UNMXR, FOREACH_SCALAR_TYPE, FOREACH_BOOLEAN_INT, APPLY_STAMP)
