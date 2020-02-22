/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HELPERS_H
#define HELPERS_H

#include <cstring>

inline size_t idx2D(const size_t i, const size_t j, const size_t lda) {
  return j * lda + i;
}

template <typename T> T machine_precision() {
  return static_cast<T>(1.19e-07); // the single precision value
}

#endif /* HELPERS_H */
