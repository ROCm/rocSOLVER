/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HELPERS_H
#define HELPERS_H
#include <cstring>

inline size_t idx2D(const size_t i, const size_t j, const size_t lda) 
{
  return j * lda + i;
}

template <typename T> 
inline T machine_precision();
template <> 
inline float machine_precision() 
{ 
    return static_cast<float>(1.19e-07); 
}
template <> 
inline double machine_precision() 
{ 
    return static_cast<double>(2.22e-16); 
}

template <typename T>
T const * cast2constType(T *array)
{
    T const *R = array;
    return R;
}

template <typename T>
T const *const * cast2constType(T *const *array)
{
    T const *const *R = array;
    return R;
}

template <typename T>
T * cast2constPointer(T *array)
{
    T *R = array;
    return R;
}

template <typename T>
T *const * cast2constPointer(T *const *array)
{
    T *const *R = array;
    return R;
}


#endif /* HELPERS_H */
