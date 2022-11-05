/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblascommon/program_options.hpp"

namespace roc
{
// Regular expression for token delimiters (whitespace and commas)
const std::regex program_options_regex{"[, \\f\\n\\r\\t\\v]+", std::regex_constants::optimize};
}
