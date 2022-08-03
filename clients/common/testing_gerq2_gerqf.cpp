
#include <testing_gerq2_gerqf.hpp>

#define TESTING_GERQ2_GERQF(...) template void testing_gerq2_gerqf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GERQ2_GERQF,
            FOREACH_BOOLEAN_0,
            FOREACH_BOOLEAN_1,
            FOREACH_BOOLEAN_INT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
