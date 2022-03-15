/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

bool set_environment_variable(const char* name, const char* value);
bool unset_environment_variable(const char* name);

class environment_error : public std::runtime_error
{
public:
    explicit environment_error(const std::string& what_arg)
        : std::runtime_error(what_arg)
    {
    }
};

class scoped_envvar
{
    std::string m_name;
    std::optional<std::string> m_old_value;

public:
    scoped_envvar(const char* name, const char* value);
    scoped_envvar(const scoped_envvar&) = delete;
    scoped_envvar(scoped_envvar&&) = delete;
    ~scoped_envvar();

    scoped_envvar& operator=(const scoped_envvar&) = delete;
    scoped_envvar& operator=(scoped_envvar&&) = delete;
};
