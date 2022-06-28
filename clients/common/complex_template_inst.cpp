#include <testing_lacgv.hpp>
#include <testing_orgbr_ungbr.hpp>
#include <testing_orglx_unglx.hpp>
#include <testing_orgtr_ungtr.hpp>
#include <testing_orgxl_ungxl.hpp>
#include <testing_orgxr_ungxr.hpp>
#include <testing_ormbr_unmbr.hpp>
#include <testing_ormlx_unmlx.hpp>
#include <testing_ormtr_unmtr.hpp>
#include <testing_ormxl_unmxl.hpp>
#include <testing_ormxr_unmxr.hpp>
#include <testing_syev_heev.hpp>
#include <testing_syevd_heevd.hpp>
#include <testing_syevx_heevx.hpp>
#include <testing_sygsx_hegsx.hpp>
#include <testing_sygv_hegv.hpp>
#include <testing_sygvd_hegvd.hpp>
#include <testing_sygvx_hegvx.hpp>
#include <testing_sytxx_hetxx.hpp>

template void testing_sygsx_hegsx<false, false, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygv_hegv<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevx_heevx<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_float_complex, 1>(Arguments& argus);
template void testing_syevd_heevd<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevd_heevd<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_syev_heev<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_float_complex, 1>(Arguments& argus);
template void testing_syevd_heevd<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevd_heevd<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_syevx_heevx<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_ormbr_unmbr<rocblas_double_complex>(Arguments& argus);
template void testing_ormbr_unmbr<rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_orglx_unglx<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orglx_unglx<rocblas_float_complex, 0>(Arguments& argus);
template void testing_orgbr_ungbr<rocblas_double_complex>(Arguments& argus);
template void testing_orgbr_ungbr<rocblas_float_complex>(Arguments& argus);
template void testing_syevx_heevx<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_syevx_heevx<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygv_hegv<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_float_complex, 0>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_orgtr_ungtr<rocblas_double_complex>(Arguments& argus);
template void testing_orgtr_ungtr<rocblas_float_complex>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_float_complex, 0>(Arguments& argus);
template void testing_sygv_hegv<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygv_hegv<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_double_complex, 0>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_float_complex, 0>(Arguments& argus);
template void testing_syev_heev<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orgxr_ungxr<rocblas_float_complex, 1>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orgxl_ungxl<rocblas_float_complex, 1>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, rocblas_float_complex>(Arguments& argus);
template void testing_ormtr_unmtr<rocblas_double_complex>(Arguments& argus);
template void testing_ormtr_unmtr<rocblas_float_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_double_complex, 1>(Arguments& argus);
template void testing_ormxr_unmxr<rocblas_float_complex, 1>(Arguments& argus);
template void testing_syev_heev<false, true, rocblas_double_complex>(Arguments& argus);
template void testing_syev_heev<false, true, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, rocblas_float_complex>(Arguments& argus);
template void testing_lacgv<rocblas_double_complex>(Arguments& argus);
template void testing_lacgv<rocblas_float_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, rocblas_double_complex>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, rocblas_float_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, rocblas_double_complex>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, rocblas_float_complex>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormxl_unmxl<rocblas_float_complex, 0>(Arguments& argus);
template void testing_orglx_unglx<rocblas_double_complex, 1>(Arguments& argus);
template void testing_orglx_unglx<rocblas_float_complex, 1>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_double_complex, 0>(Arguments& argus);
template void testing_ormlx_unmlx<rocblas_float_complex, 0>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, rocblas_double_complex>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, rocblas_float_complex>(Arguments& argus);
