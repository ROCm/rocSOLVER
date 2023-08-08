
% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_20_60';
[T,Q] = regen_prob(dir);
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_20_100';
[T,Q] = regen_prob(dir);
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_20_140';
[T,Q] = regen_prob(dir);
