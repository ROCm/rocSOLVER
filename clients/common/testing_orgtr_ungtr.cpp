
#include <testing_orgtr_ungtr.hpp>

#include <client_util.hpp>

#define TESTING_ORGTR_UNGTR(...) template void testing_orgtr_ungtr<__VA_ARGS__>(Arguments&);
INSTANTIATE(TESTING_ORGTR_UNGTR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
