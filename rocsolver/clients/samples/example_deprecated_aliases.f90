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

subroutine ROCSOLVER_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocsolver error'
        stop
    endif
end subroutine ROCSOLVER_CHECK

module rocsolver_enums
    use iso_c_binding

    !!!!!!!!!!!!!!!!!!!!!!!
    !    rocSOLVER types    !
    !!!!!!!!!!!!!!!!!!!!!!!
    enum, bind(c)
      enumerator :: rocsolver_operation_none = 111
        enumerator :: rocsolver_operation_transpose = 112
        enumerator :: rocsolver_operation_conjugate_transpose = 113
    end enum

    enum, bind(c)
        enumerator :: rocsolver_fill_upper = 121
        enumerator :: rocsolver_fill_lower = 122
        enumerator :: rocsolver_fill_full  = 123
    end enum

    enum, bind(c)
        enumerator :: rocsolver_diagonal_non_unit = 131
        enumerator :: rocsolver_diagonal_unit     = 132
    end enum

    enum, bind(c)
        enumerator :: rocsolver_side_left  = 141
        enumerator :: rocsolver_side_right = 142
        enumerator :: rocsolver_side_both  = 143
    end enum

    enum, bind(c)
        enumerator :: rocsolver_status_success         = 0
        enumerator :: rocsolver_status_invalid_handle  = 1
        enumerator :: rocsolver_status_not_implemented = 2
        enumerator :: rocsolver_status_invalid_pointer = 3
        enumerator :: rocsolver_status_invalid_size    = 4
        enumerator :: rocsolver_status_memory_error    = 5
        enumerator :: rocsolver_status_internal_error  = 6
        enumerator :: rocsolver_status_perf_degraded   = 7
        enumerator :: rocsolver_status_size_query_mismatch = 8
        enumerator :: rocsolver_status_size_increased      = 9
        enumerator :: rocsolver_status_size_unchanged      = 10
        enumerator :: rocsolver_status_invalid_value       = 11
        enumerator :: rocsolver_status_continue            = 12
    end enum

end module rocsolver_enums

module rocsolver
    use iso_c_binding

    interface
        function rocsolver_create_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocsolver_create_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocsolver_create_handle
    end interface

    interface
        function rocsolver_destroy_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocsolver_destroy_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocsolver_destroy_handle
    end interface

end module rocsolver

program example_deprecated_aliases
    use iso_c_binding
    use rocblas
    use rocsolver

    implicit none

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

    ! Create rocBLAS handle using the deprecated rocSOLVER function
    type(c_ptr), target :: handle
    call ROCSOLVER_CHECK(rocsolver_create_handle(c_loc(handle)))

    call ROCSOLVER_CHECK(rocsolver_destroy_handle(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_deprecated_aliases
