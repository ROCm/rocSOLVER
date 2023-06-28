hipcc   \
	-fopenmp \
	-I /opt/rocm/include \
	-I /opt/rocm/include/rocblas  \
	-I ../include \
        -I /home/edazeved/MERGE/rocSOLVER/common/include/ \
	-D__HIP_PLATFORM_AMD__ \
	-Wall -ftrapv \
	--offload-arch=gfx1030 -std=c++17    \
	-c \
	$*

#        -I /home/edazeved/WORK/rocBLAS/build/release/rocblas-install/rocblas \
#	-I /home/edazeved/WORK/rocBLAS/build/release/rocblas-install/include/  \
