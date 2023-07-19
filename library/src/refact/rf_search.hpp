
/************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

// -------------------------------------------------
// function to perform search in array
// -------------------------------------------------
// search array ind[istart], ..., ind[iend-1]
// for matching value "key"
//
// return the index value of matching position
// ---------------------------------------
auto rf_search = [] (auto ind[], auto istart, auto iend, auto key) 
{
    // -----------------
    // use binary search
    // -----------------

    auto const small_size = 8;
    while(iend - istart > small_size)
    {
        auto const imid = istart + (iend - istart) / 2;
        auto const curr = ind[imid];

        if(curr == key)
            return imid;
        else if(curr > key)
            iend = imid;
        else
            istart = imid + 1;
    };

// ------------------------
// use simple linear search
// ------------------------
#pragma unroll 4
    for(auto imid = istart; imid < iend; imid++)
    {
        if(ind[imid] == key)
            return imid;
    };

    return -1;
};
