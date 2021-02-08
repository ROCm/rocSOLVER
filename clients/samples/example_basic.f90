!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if
end subroutine HIP_CHECK

subroutine ROCBLAS_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocblas error'
        stop
    endif
end subroutine ROCBLAS_CHECK

subroutine ROCSOLVER_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocsolver error'
        stop
    endif
end subroutine ROCSOLVER_CHECK

program example_fortran_basic
    use iso_c_binding
    use rocblas

    implicit none

    interface
        function rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv) &
                result(c_int) &
                bind(c, name = 'rocsolver_dgeqrf')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: M
            integer(c_int), value :: N
            type(c_ptr), value :: dA
            integer(c_int), value :: lda
            type(c_ptr), value :: dIpiv
        end function rocsolver_dgeqrf
    end interface

    ! TODO: hip workaround until plugin is ready.
    interface
        function hipMalloc(ptr, size) &
                result(c_int) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc
    end interface

    interface
        function hipFree(ptr) &
                result(c_int) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
        end function hipFree
    end interface

    interface
        function hipMemcpy(dst, src, size, kind) &
                result(c_int) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    interface
        function hipMemset(dst, val, size) &
                result(c_int) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset
    end interface

    interface
        function hipDeviceSynchronize() &
                result(c_int) &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
        end function hipDeviceSynchronize
    end interface

    interface
        function hipDeviceReset() &
                result(c_int) &
                bind(c, name = 'hipDeviceReset')
            use iso_c_binding
            implicit none
        end function hipDeviceReset
    end interface
    ! TODO end

    integer :: i, j ! indices for iterating over results

    ! Define our input data
    real(c_double), target :: hA(3,3) = reshape((/12, 6, -4, -51, 167, 24, 4, -68, -41/), (/3, 3/))
    integer(c_int), parameter :: M = 3
    integer(c_int), parameter :: N = 3
    integer(c_int), parameter :: lda = 3

    real(c_double), target :: hIpiv(3) ! CPU buffer for Householder scalars
    integer(c_size_t) :: size_A = size(hA)
    integer(c_size_t) :: size_Ipiv = size(hIpiv)

    type(c_ptr), target :: dA     ! GPU buffer for A
    type(c_ptr), target :: dIpiv  ! GPU buffer for Householder scalars

    type(c_ptr), target :: handle ! rocblas_handle

    ! Allocate device-side memory
    call HIP_CHECK(hipMalloc(c_loc(dA), size_A * 8))
    call HIP_CHECK(hipMalloc(c_loc(dIpiv), size_Ipiv * 8))

    ! Create rocBLAS handle
    call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

    ! Copy memory from host to device
    call HIP_CHECK(hipMemcpy(dA, c_loc(hA), size_A * 8, 1))

    ! Compute the QR factorization on the device
    call ROCSOLVER_CHECK(rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv))

    ! Copy result from device to host
    call HIP_CHECK(hipMemcpy(c_loc(hA), dA, size_A * 8, 2))
    call HIP_CHECK(hipMemcpy(c_loc(hIpiv), dIpiv, size_Ipiv * 8, 2))

    ! Output results
    do i = 1,size(hA,1)
      print *, (hA(i,j), j=1,size(hA,2))
    end do

    ! Clean up
    call HIP_CHECK(hipFree(dA))
    call HIP_CHECK(hipFree(dIpiv))
    call ROCSOLVER_CHECK(rocblas_destroy_handle(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_fortran_basic
