/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdlib>
#include <system_error>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "rocblascommon/clients_utility.hpp"
#include "rocsolver_test.hpp"

fs::path get_sparse_data_dir()
{
    // first check an environment variable
    if(const char* datadir = std::getenv("ROCSOLVER_TEST_DATA"))
        return fs::path{datadir};

    std::vector<std::string> considered;

    // check relative to the running executable
    fs::path exe_path = fs::path(rocsolver_exepath());
    std::vector<fs::path> candidates = {"../share/rocsolver/test", "sparsedata"};
    for(const fs::path& candidate : candidates)
    {
        std::error_code ec;
        fs::path exe_relative = fs::canonical(exe_path / candidate, ec);
        if(!ec)
            return exe_relative;
        considered.push_back(exe_relative.string());
    }

    fmt::print(stderr,
               "Warning: default sparse data directories not found. "
               "Defaulting to current working directory.\nExecutable location: {}\n"
               "Paths considered:\n{}\n",
               exe_path.string(), fmt::join(considered, "\n"));

    return fs::current_path();
}
