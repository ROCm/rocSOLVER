#include <testing_bdsqr.hpp>
#include <testing_gebd2_gebrd.hpp>
#include <testing_gelq2_gelqf.hpp>
#include <testing_gels.hpp>
#include <testing_geql2_geqlf.hpp>
#include <testing_geqr2_geqrf.hpp>
#include <testing_gerq2_gerqf.hpp>
#include <testing_gesv.hpp>
#include <testing_gesvd.hpp>
#include <testing_getf2_getrf.hpp>
#include <testing_getf2_getrf_npvt.hpp>
#include <testing_getri.hpp>
#include <testing_getri_npvt.hpp>
#include <testing_getri_npvt_outofplace.hpp>
#include <testing_getri_outofplace.hpp>
#include <testing_getrs.hpp>
#include <testing_labrd.hpp>
#include <testing_larf.hpp>
#include <testing_larfb.hpp>
#include <testing_larfg.hpp>
#include <testing_larft.hpp>
#include <testing_laswp.hpp>
#include <testing_lasyf.hpp>
#include <testing_latrd.hpp>
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
#include <testing_posv.hpp>
#include <testing_potf2_potrf.hpp>
#include <testing_potri.hpp>
#include <testing_potrs.hpp>
#include <testing_stebz.hpp>
#include <testing_stedc.hpp>
#include <testing_stein.hpp>
#include <testing_steqr.hpp>
#include <testing_sterf.hpp>
#include <testing_syev_heev.hpp>
#include <testing_syevd_heevd.hpp>
#include <testing_syevx_heevx.hpp>
#include <testing_sygsx_hegsx.hpp>
#include <testing_sygv_hegv.hpp>
#include <testing_sygvd_hegvd.hpp>
#include <testing_sygvx_hegvx.hpp>
#include <testing_sytf2_sytrf.hpp>
#include <testing_sytxx_hetxx.hpp>
#include <testing_trtri.hpp>

template void testing_sygsx_hegsx<false, false, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 0, float>(Arguments& argus);
template void testing_getrs<false, true, double>(Arguments& argus);
template void testing_getrs<false, true, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 0, float>(Arguments& argus);
template void testing_getf2_getrf<false, false, 0, double>(Arguments& argus);
template void testing_getf2_getrf<false, false, 0, float>(Arguments& argus);
template void testing_gels<true, true, double>(Arguments& argus);
template void testing_gels<true, true, float>(Arguments& argus);
template void testing_orgxr_ungxr<double, 1>(Arguments& argus);
template void testing_orgxr_ungxr<float, 1>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 0, float>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 0, float>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 0, float>(Arguments& argus);
template void testing_syevx_heevx<false, true, double>(Arguments& argus);
template void testing_syevx_heevx<false, true, float>(Arguments& argus);
template void testing_syevx_heevx<true, true, double>(Arguments& argus);
template void testing_syevx_heevx<true, true, float>(Arguments& argus);
template void testing_gesvd<false, true, double>(Arguments& argus);
template void testing_gesvd<false, true, float>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<false, false, 1, float>(Arguments& argus);
template void testing_gesv<false, false, double>(Arguments& argus);
template void testing_gesv<false, false, float>(Arguments& argus);
template void testing_getri_npvt_outofplace<true, true, double>(Arguments& argus);
template void testing_getri_npvt_outofplace<true, true, float>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 0, float>(Arguments& argus);
template void testing_gesvd<true, true, double>(Arguments& argus);
template void testing_gesvd<true, true, float>(Arguments& argus);
template void testing_potf2_potrf<false, true, 1, double>(Arguments& argus);
template void testing_potf2_potrf<false, true, 1, float>(Arguments& argus);
template void testing_sterf<double>(Arguments& argus);
template void testing_sterf<float>(Arguments& argus);
template void testing_getf2_getrf<true, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf<true, true, 1, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 0, float>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 0, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 1, float>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, false, 1, float>(Arguments& argus);
template void testing_getri<false, true, double>(Arguments& argus);
template void testing_getri<false, true, float>(Arguments& argus);
template void testing_bdsqr<double>(Arguments& argus);
template void testing_bdsqr<float>(Arguments& argus);
template void testing_ormlx_unmlx<double, 1>(Arguments& argus);
template void testing_ormlx_unmlx<float, 1>(Arguments& argus);
template void testing_stein<double>(Arguments& argus);
template void testing_stein<float>(Arguments& argus);
template void testing_getri_npvt_outofplace<false, true, double>(Arguments& argus);
template void testing_getri_npvt_outofplace<false, true, float>(Arguments& argus);
template void testing_orgbr_ungbr<double>(Arguments& argus);
template void testing_orgbr_ungbr<float>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, true, 0, float>(Arguments& argus);
template void testing_orgxl_ungxl<double, 1>(Arguments& argus);
template void testing_orgxl_ungxl<float, 1>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<true, true, 1, float>(Arguments& argus);
template void testing_potri<true, true, double>(Arguments& argus);
template void testing_potri<true, true, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 0, float>(Arguments& argus);
template void testing_getri_outofplace<false, false, double>(Arguments& argus);
template void testing_getri_outofplace<false, false, float>(Arguments& argus);
template void testing_orgxr_ungxr<double, 0>(Arguments& argus);
template void testing_orgxr_ungxr<float, 0>(Arguments& argus);
template void testing_potri<false, true, double>(Arguments& argus);
template void testing_potri<false, true, float>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 1, float>(Arguments& argus);
template void testing_ormxl_unmxl<double, 1>(Arguments& argus);
template void testing_ormxl_unmxl<float, 1>(Arguments& argus);
template void testing_sygv_hegv<false, true, double>(Arguments& argus);
template void testing_sygv_hegv<false, true, float>(Arguments& argus);
template void testing_labrd<double>(Arguments& argus);
template void testing_labrd<float>(Arguments& argus);
template void testing_orgxl_ungxl<double, 0>(Arguments& argus);
template void testing_orgxl_ungxl<float, 0>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<true, true, 0, float>(Arguments& argus);
template void testing_getri_npvt<true, true, double>(Arguments& argus);
template void testing_getri_npvt<true, true, float>(Arguments& argus);
template void testing_ormtr_unmtr<double>(Arguments& argus);
template void testing_ormtr_unmtr<float>(Arguments& argus);
template void testing_potf2_potrf<true, true, 1, double>(Arguments& argus);
template void testing_potf2_potrf<true, true, 1, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, false, 1, float>(Arguments& argus);
template void testing_potf2_potrf<true, true, 0, double>(Arguments& argus);
template void testing_potf2_potrf<true, true, 0, float>(Arguments& argus);
template void testing_getri_npvt<false, false, double>(Arguments& argus);
template void testing_getri_npvt<false, false, float>(Arguments& argus);
template void testing_potrs<false, false, double>(Arguments& argus);
template void testing_potrs<false, false, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 1, float>(Arguments& argus);
template void testing_orgtr_ungtr<double>(Arguments& argus);
template void testing_orgtr_ungtr<float>(Arguments& argus);
template void testing_getf2_getrf<false, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf<false, true, 1, float>(Arguments& argus);
template void testing_posv<true, true, double>(Arguments& argus);
template void testing_posv<true, true, float>(Arguments& argus);
template void testing_getf2_getrf<false, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf<false, true, 0, float>(Arguments& argus);
template void testing_gels<false, false, double>(Arguments& argus);
template void testing_gels<false, false, float>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, double>(Arguments& argus);
template void testing_sygvx_hegvx<false, false, float>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, double>(Arguments& argus);
template void testing_sygvx_hegvx<false, true, float>(Arguments& argus);
template void testing_getrs<false, false, double>(Arguments& argus);
template void testing_getrs<false, false, float>(Arguments& argus);
template void testing_geqr2_geqrf<true, false, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, false, 1, float>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 0, float>(Arguments& argus);
template void testing_larft<double>(Arguments& argus);
template void testing_larft<float>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, false, 1, float>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, double>(Arguments& argus);
template void testing_sygvx_hegvx<true, true, float>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 0, float>(Arguments& argus);
template void testing_getf2_getrf<true, true, 0, double>(Arguments& argus);
template void testing_getf2_getrf<true, true, 0, float>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, double>(Arguments& argus);
template void testing_sygvd_hegvd<false, false, float>(Arguments& argus);
template void testing_trtri<false, true, double>(Arguments& argus);
template void testing_trtri<false, true, float>(Arguments& argus);
template void testing_ormbr_unmbr<double>(Arguments& argus);
template void testing_ormbr_unmbr<float>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<true, true, 1, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 1, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 1, float>(Arguments& argus);
template void testing_larfb<double>(Arguments& argus);
template void testing_larfb<float>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, double>(Arguments& argus);
template void testing_sytxx_hetxx<false, true, 1, float>(Arguments& argus);
template void testing_syev_heev<false, false, double>(Arguments& argus);
template void testing_syev_heev<false, false, float>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, double>(Arguments& argus);
template void testing_sygvd_hegvd<false, true, float>(Arguments& argus);
template void testing_potf2_potrf<false, true, 0, double>(Arguments& argus);
template void testing_potf2_potrf<false, true, 0, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 1, float>(Arguments& argus);
template void testing_gesv<false, true, double>(Arguments& argus);
template void testing_gesv<false, true, float>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<false, true, 0, float>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 1, double>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 1, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 1, float>(Arguments& argus);
template void testing_getri_outofplace<false, true, double>(Arguments& argus);
template void testing_getri_outofplace<false, true, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 0, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 0, float>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 1, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, false, 1, float>(Arguments& argus);
template void testing_getri_npvt_outofplace<false, false, double>(Arguments& argus);
template void testing_getri_npvt_outofplace<false, false, float>(Arguments& argus);
template void testing_ormxr_unmxr<double, 1>(Arguments& argus);
template void testing_ormxr_unmxr<float, 1>(Arguments& argus);
template void testing_posv<false, true, double>(Arguments& argus);
template void testing_posv<false, true, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, false, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<true, true, 1, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 0, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 0, float>(Arguments& argus);
template void testing_getrs<true, true, double>(Arguments& argus);
template void testing_getrs<true, true, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 1, float>(Arguments& argus);
template void testing_syevx_heevx<false, false, double>(Arguments& argus);
template void testing_syevx_heevx<false, false, float>(Arguments& argus);
template void testing_stedc<double>(Arguments& argus);
template void testing_stedc<float>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, double>(Arguments& argus);
template void testing_sygsx_hegsx<true, true, 0, float>(Arguments& argus);
template void testing_larfg<double>(Arguments& argus);
template void testing_larfg<float>(Arguments& argus);
template void testing_potrs<false, true, double>(Arguments& argus);
template void testing_potrs<false, true, float>(Arguments& argus);
template void testing_potrs<true, true, double>(Arguments& argus);
template void testing_potrs<true, true, float>(Arguments& argus);
template void testing_getri_npvt<false, true, double>(Arguments& argus);
template void testing_getri_npvt<false, true, float>(Arguments& argus);
template void testing_getf2_getrf<false, false, 1, double>(Arguments& argus);
template void testing_getf2_getrf<false, false, 1, float>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 0, double>(Arguments& argus);
template void testing_gerq2_gerqf<false, true, 0, float>(Arguments& argus);
template void testing_potf2_potrf<false, false, 0, double>(Arguments& argus);
template void testing_potf2_potrf<false, false, 0, float>(Arguments& argus);
template void testing_posv<false, false, double>(Arguments& argus);
template void testing_posv<false, false, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 1, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, true, 1, float>(Arguments& argus);
template void testing_laswp<double>(Arguments& argus);
template void testing_laswp<float>(Arguments& argus);
template void testing_latrd<double>(Arguments& argus);
template void testing_latrd<float>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<true, true, 1, float>(Arguments& argus);
template void testing_gels<false, true, double>(Arguments& argus);
template void testing_gels<false, true, float>(Arguments& argus);
template void testing_ormxr_unmxr<double, 0>(Arguments& argus);
template void testing_ormxr_unmxr<float, 0>(Arguments& argus);
template void testing_syevd_heevd<false, false, double>(Arguments& argus);
template void testing_syevd_heevd<false, false, float>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 0, double>(Arguments& argus);
template void testing_gebd2_gebrd<false, true, 0, float>(Arguments& argus);
template void testing_sygv_hegv<true, true, double>(Arguments& argus);
template void testing_sygv_hegv<true, true, float>(Arguments& argus);
template void testing_ormlx_unmlx<double, 0>(Arguments& argus);
template void testing_ormlx_unmlx<float, 0>(Arguments& argus);
template void testing_getri_outofplace<true, true, double>(Arguments& argus);
template void testing_getri_outofplace<true, true, float>(Arguments& argus);
template void testing_sygv_hegv<false, false, double>(Arguments& argus);
template void testing_sygv_hegv<false, false, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 1, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, true, 1, float>(Arguments& argus);
template void testing_lasyf<double>(Arguments& argus);
template void testing_lasyf<float>(Arguments& argus);
template void testing_syevd_heevd<false, true, double>(Arguments& argus);
template void testing_syevd_heevd<false, true, float>(Arguments& argus);
template void testing_trtri<false, false, double>(Arguments& argus);
template void testing_trtri<false, false, float>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, double>(Arguments& argus);
template void testing_sygvd_hegvd<true, true, float>(Arguments& argus);
template void testing_orglx_unglx<double, 1>(Arguments& argus);
template void testing_orglx_unglx<float, 1>(Arguments& argus);
template void testing_orglx_unglx<double, 0>(Arguments& argus);
template void testing_orglx_unglx<float, 0>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 1, double>(Arguments& argus);
template void testing_gebd2_gebrd<true, true, 1, float>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<false, false, 0, float>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 1, double>(Arguments& argus);
template void testing_getf2_getrf_npvt<false, true, 1, float>(Arguments& argus);
template void testing_larf<double>(Arguments& argus);
template void testing_larf<float>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 0, double>(Arguments& argus);
template void testing_geql2_geqlf<false, true, 0, float>(Arguments& argus);
template void testing_syevd_heevd<true, true, double>(Arguments& argus);
template void testing_syevd_heevd<true, true, float>(Arguments& argus);
template void testing_ormxl_unmxl<double, 0>(Arguments& argus);
template void testing_ormxl_unmxl<float, 0>(Arguments& argus);
template void testing_stebz<double>(Arguments& argus);
template void testing_stebz<float>(Arguments& argus);
template void testing_syev_heev<false, true, double>(Arguments& argus);
template void testing_syev_heev<false, true, float>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 0, double>(Arguments& argus);
template void testing_sytf2_sytrf<true, true, 0, float>(Arguments& argus);
template void testing_potf2_potrf<false, false, 1, double>(Arguments& argus);
template void testing_potf2_potrf<false, false, 1, float>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, double>(Arguments& argus);
template void testing_sytxx_hetxx<true, true, 0, float>(Arguments& argus);
template void testing_syev_heev<true, true, double>(Arguments& argus);
template void testing_syev_heev<true, true, float>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 1, double>(Arguments& argus);
template void testing_geqr2_geqrf<false, false, 1, float>(Arguments& argus);
template void testing_gesvd<false, false, double>(Arguments& argus);
template void testing_gesvd<false, false, float>(Arguments& argus);
template void testing_trtri<true, true, double>(Arguments& argus);
template void testing_trtri<true, true, float>(Arguments& argus);
template void testing_getri<true, true, double>(Arguments& argus);
template void testing_getri<true, true, float>(Arguments& argus);
template void testing_potri<false, false, double>(Arguments& argus);
template void testing_potri<false, false, float>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 0, double>(Arguments& argus);
template void testing_gelq2_gelqf<false, false, 0, float>(Arguments& argus);
template void testing_getri<false, false, double>(Arguments& argus);
template void testing_getri<false, false, float>(Arguments& argus);
template void testing_gesv<true, true, double>(Arguments& argus);
template void testing_gesv<true, true, float>(Arguments& argus);
template void testing_steqr<double>(Arguments& argus);
template void testing_steqr<float>(Arguments& argus);
