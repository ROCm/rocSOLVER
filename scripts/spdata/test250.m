% ********************************************************************
% Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
% ********************************************************************
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_250_300';
[T,Q] = regen_prob(dir);
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_250_500';
[T,Q] = regen_prob(dir);
dir='/home/edazeved/MERGE/rocSOLVER/build/debug/clients/sparsedata/posmat_250_700';
[T,Q] = regen_prob(dir);
