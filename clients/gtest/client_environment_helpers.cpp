/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "client_environment_helpers.hpp"

bool set_environment_variable(const char* name, const char* value)
{
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

bool unset_environment_variable(const char* name)
{
#ifdef _WIN32
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

scoped_envvar::scoped_envvar(const char* name, const char* value)
    : m_name(name)
{
    if(const char* old_value = std::getenv(name))
    {
        m_old_value.emplace(old_value);
    }
    if(!set_environment_variable(name, value))
        throw environment_error(fmt::format("failed to set {:s}={:s}", name, value));
}

scoped_envvar::~scoped_envvar()
{
    if(m_old_value)
        set_environment_variable(m_name.c_str(), m_old_value->c_str());
    else
        unset_environment_variable(m_name.c_str());
}
