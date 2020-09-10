/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_TEST_H_
#define ROCBLAS_TEST_H_

//#include "argument_model.hpp"
#include "rocblas.h"
//#include "rocblas_arguments.hpp"
//#include "test_cleanup.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "../../library/src/include/rocblas_ostream.hpp"

// Suppress warnings about hipMalloc(), hipFree() except in rocblas-test and
// rocblas-bench
#if !defined(GOOGLE_TEST) && !defined(ROCBLAS_BENCH)
#undef hipMalloc
#undef hipFree
#endif

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>

// Extra macro so that macro arguments get expanded before calling Google Test
#define CHECK_HIP_ERROR2(ERROR) ASSERT_EQ(ERROR, hipSuccess)
#define CHECK_HIP_ERROR(ERROR) CHECK_HIP_ERROR2(ERROR)

#define CHECK_DEVICE_ALLOCATION(ERROR)                                         \
  do {                                                                         \
    auto error = ERROR;                                                        \
    if (error == hipErrorOutOfMemory) {                                        \
      SUCCEED() << LIMITED_MEMORY_STRING;                                      \
      return;                                                                  \
    } else if (error != hipSuccess) {                                          \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define EXPECT_ROCBLAS_STATUS ASSERT_EQ

#else // GOOGLE_TEST

inline void rocblas_expect_status(rocblas_status status,
                                  rocblas_status expect) {
  if (status != expect) {
    rocblas_cerr << "rocBLAS status error: Expected "
                 << rocblas_status_to_string(expect) << ", received "
                 << rocblas_status_to_string(status) << std::endl;
    if (expect == rocblas_status_success)
      exit(EXIT_FAILURE);
  }
}

#define CHECK_HIP_ERROR(ERROR)                                                 \
  do {                                                                         \
    auto error = ERROR;                                                        \
    if (error != hipSuccess) {                                                 \
      rocblas_cerr << "error: " << hipGetErrorString(error) << " (" << error   \
                   << ") at " __FILE__ ":" << __LINE__ << std::endl;           \
      rocblas_abort();                                                         \
    }                                                                          \
  } while (0)

#define CHECK_DEVICE_ALLOCATION(ERROR)

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#endif // GOOGLE_TEST

#define CHECK_ROCBLAS_ERROR2(STATUS)                                           \
  EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

/*
#ifdef GOOGLE_TEST

// Function which matches Arguments with a category, accounting for
arg.known_bug_platforms bool match_test_category(const Arguments& arg, const
char* category);

// The tests are instantiated by filtering through the RocBLAS_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, categ0ry) \
    INSTANTIATE_TEST_SUITE_P(categ0ry, \
                            testclass, \
                            testing::ValuesIn(RocBLAS_TestData::begin([](const
Arguments& arg) { \
                                                  return
testclass::type_filter(arg)             \
                                                         &&
testclass::function_filter(arg)      \
                                                         &&
match_test_category(arg, #categ0ry); \
                                              }), \
                                              RocBLAS_TestData::end()), \
                            testclass::PrintToStringParamName());

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)        \
    INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
    INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
    INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
    INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

// Function to catch signals and exceptions as failures
void catch_signals_and_exceptions_as_failures(const std::function<void()>&
test);

// Macro to call catch_signals_and_exceptions_as_failures() with a lambda
expression
#define CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(test) \
    catch_signals_and_exceptions_as_failures([&] { test; })

// Template parameter is used to generate multiple instantiations
template <typename>
class RocBLAS_TestName
{
    std::ostringstream str;

    static auto& get_table()
    {
        // Placed inside function to avoid dependency on initialization order
        static std::unordered_map<std::string, size_t>* table =
test_cleanup::allocate(&table); return *table;
    }

public:
    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantation of RocBLAS_TestName
        auto&       table = get_table();
        std::string name(str.str());

        if(name == "")
            name = "1";

        // Warn about unset letter parameters
        if(name.find('*') != name.npos)
            rocblas_cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                            "Warning: Character * found in name."
                            " This means a required letter parameter\n"
                            "(e.g., transA, diag, etc.) has not been set in the
YAML file." " Check the YAML file.\n"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                         << std::endl;

        // Replace non-alphanumeric characters with letters
        std::replace(name.begin(), name.end(), '-', 'n'); // minus
        std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

        // Complex (A,B) is replaced with ArBi
        name.erase(std::remove(name.begin(), name.end(), '('), name.end());
        std::replace(name.begin(), name.end(), ',', 'r');
        std::replace(name.begin(), name.end(), ')', 'i');

        // If parameters are repeated, append an incrementing suffix
        auto p = table.find(name);
        if(p != table.end())
            name += "_t" + std::to_string(++p->second);
        else
            table[name] = 1;

        return name;
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend RocBLAS_TestName& operator<<(RocBLAS_TestName& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend RocBLAS_TestName&& operator<<(RocBLAS_TestName&& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return std::move(name);
    }

    RocBLAS_TestName()                        = default;
    RocBLAS_TestName(const RocBLAS_TestName&) = delete;
    RocBLAS_TestName& operator=(const RocBLAS_TestName&) = delete;
};

// ----------------------------------------------------------------------------
// RocBLAS_Test base class. All non-legacy rocBLAS Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class RocBLAS_Test : public testing::TestWithParam<Arguments>
{
protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments&)
        {
            return static_cast<bool>(FILTER<T...>{});
        }
    };

public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info)
const
        {
            return TEST::name_suffix(info.param);
        }
    };
};

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Normal tests which return true when converted to bool
// ----------------------------------------------------------------------------
struct rocblas_test_valid
{
    // Return true to indicate the type combination is valid, for filtering
    virtual explicit operator bool() final
    {
        return true;
    }

    // Require derived class to define functor which takes (const Arguments &)
    virtual void operator()(const Arguments&) = 0;
};

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct rocblas_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    virtual explicit operator bool() final
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    virtual void operator()(const Arguments&) final
    {
        static constexpr char msg[] = "Internal error: Test called with invalid
types";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        rocblas_cerr << msg << std::endl;
        rocblas_abort();
#endif
    }
};
*/

#endif
