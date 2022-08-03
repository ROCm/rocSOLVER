
#include <testing_ormlx_unmlx.hpp>

#define TESTING_ORMLX_UNMLX(...) template void testing_ormlx_unmlx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMLX_UNMLX, FOREACH_SCALAR_TYPE, FOREACH_BOOLEAN_INT, APPLY_STAMP)
