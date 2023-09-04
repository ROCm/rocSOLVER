
/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

// -------------------------------------------------
// function to perform search in array
// -------------------------------------------------
// search array ind[istart], ..., ind[iend-1]
// for matching value "key"
//
// return the index value of matching position
// ---------------------------------------
template <typename T>
__device__ rocblas_int rf_search(rocblas_int* ind, rocblas_int istart, rocblas_int iend, rocblas_int key)
{
    // -----------------
    // use binary search
    // -----------------
    rocblas_int const small_size = 8;
    while(iend - istart > small_size)
    {
        rocblas_int imid = istart + (iend - istart) / 2;
        rocblas_int curr = ind[imid];

        if(curr == key)
            return imid;
        else if(curr > key)
            iend = imid;
        else
            istart = imid + 1;
    }

    // ------------------------
    // use simple linear search
    // ------------------------
    for(rocblas_int imid = istart; imid < iend; imid++)
    {
        if(ind[imid] == key)
            return imid;
    }

    return -1;
}

// ------------------------------------------------------------
// Compute the inverse permutation inv_pivQ[] from pivQ
// ------------------------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void rf_ipvec_kernel(rocblas_int n, rocblas_int* pivQ, rocblas_int* inv_pivQ)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        rocblas_int iold = pivQ[tid];
        inv_pivQ[iold] = tid;
    }
}
