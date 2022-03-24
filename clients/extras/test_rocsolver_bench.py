# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
# ########################################################################

import collections
import os
import re
import shlex
import unittest
from subprocess import Popen, PIPE

def call_rocsolver_bench(*args):
    cmd = ['./rocsolver-bench']
    for arg in args:
        if isinstance(arg, str):
            cmd.extend(shlex.split(arg, False, False))
        elif isinstance(arg, collections.Sequence):
            cmd.extend(arg)
        else:
            cmd.push(str(arg))
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return (str(stdout, encoding='utf-8', errors='surrogateescape'),
            str(stderr, encoding='utf-8', errors='surrogateescape'),
            process.returncode)

class TestRocsolverBench(unittest.TestCase):
    def parse_arguments(self, bench_output):
        m = re.search(r"\n=+\nArguments:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
                bench_output, re.MULTILINE)
        self.assertTrue(m)
        arg_names = m.group('arg_names').split()
        arg_values = m.group('arg_values').split()
        self.assertEqual(len(arg_names), len(arg_values))
        return dict(zip(arg_names, arg_values))

    def parse_results(self, bench_output):
        m = re.search(r"\n=+\nResults:\s*\n=+\n(?P<arg_names>.*)\n(?P<arg_values>.*)\n",
                bench_output, re.MULTILINE)
        self.assertTrue(m)
        arg_names = m.group('arg_names').split()
        arg_values = m.group('arg_values').split()
        self.assertEqual(len(arg_names), len(arg_values))
        return dict(zip(arg_names, arg_values))

    def test_help(self):
        out, err, exitcode = call_rocsolver_bench('--help')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)
        self.assertNotEqual(out, '')

    def test_validate_precision(self):
        for precision in 'sdcz':
            with self.subTest(precision=precision):
                out, err, exitcode = call_rocsolver_bench(f'-f gels -r {precision} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gels -r 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_operation(self):
        for trans in 'NTC':
            with self.subTest(trans=trans):
                out, err, exitcode = call_rocsolver_bench(f'-f gels --trans {trans} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gels --trans 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_side(self):
        for side in 'LRB':
            with self.subTest(side=side):
                out, err, exitcode = call_rocsolver_bench(f'-f larf --side {side} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larf --side 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_fill(self):
        for fill in 'ULF':
            with self.subTest(fill=fill):
                out, err, exitcode = call_rocsolver_bench(f'-f bdsqr --uplo {fill} -n 15 --nu 10 --nv 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f bdsqr --uplo 0 -n 15 --nu 10 --nv 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_diagonal(self):
        for diag in 'NU':
            with self.subTest(diag=diag):
                out, err, exitcode = call_rocsolver_bench(f'-f trtri --diag {diag} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f trtri --diag 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_direct(self):
        for direct in 'FB':
            with self.subTest(direct=direct):
                out, err, exitcode = call_rocsolver_bench(f'-f larfb --direct {direct} --side L --storev C -n 10 -m 15 -k 1')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larfb --direct 0 --side L --storev C -k 1 -n 10 -m 15 -k 1')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_storev(self):
        for storev in 'CR':
            with self.subTest(storev=storev):
                out, err, exitcode = call_rocsolver_bench(f'-f larft --storev {storev} -n 10 -k 1')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f larft --storev 0 -n 10 -k 1')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_left_svect(self):
        for svect in 'ASON':
            with self.subTest(svect=svect):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --left_svect {svect} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --left_svect 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_right_svect(self):
        for svect in 'ASON':
            with self.subTest(svect=svect):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --right_svect {svect} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --right_svect 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_workmode(self):
        for workmode in 'IO':
            with self.subTest(workmode=workmode):
                out, err, exitcode = call_rocsolver_bench(f'-f gesvd --fast_alg {workmode} -n 10 -m 15')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f gesvd --fast_alg 0 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_evect(self):
        for evect in 'VIN':
            with self.subTest(evect=evect):
                out, err, exitcode = call_rocsolver_bench(f'-f syev --evect {evect} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f syev --evect 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_erange(self):
        for erange in 'AVI':
            with self.subTest(erange=erange):
                out, err, exitcode = call_rocsolver_bench(f'-f stebz --range {erange} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f stebz --range 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_eorder(self):
        for eorder in 'BE':
            with self.subTest(eorder=eorder):
                out, err, exitcode = call_rocsolver_bench(f'-f stebz --order {eorder} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f stebz --order 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_validate_itype(self):
        for itype in '123':
            with self.subTest(itype=itype):
                out, err, exitcode = call_rocsolver_bench(f'-f sygv --itype {itype} -n 10')
                self.assertEqual(err, '')
                self.assertEqual(exitcode, 0)

        out, err, exitcode = call_rocsolver_bench('-f sygv --itype 0 -n 10')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_unused_arg(self):
        out, err, exitcode = call_rocsolver_bench('-f gels --itype 1 -n 10 -m 15')
        self.assertNotEqual(err, '')
        self.assertNotEqual(exitcode, 0)

    def test_perf_returns_number(self):
        out, err, exitcode = call_rocsolver_bench('-f gels -n 10 -m 15 --perf 1')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)
        self.assertGreaterEqual(float(out), 0)

    def test_gels(self):
        out, err, exitcode = call_rocsolver_bench('-f gels -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gels_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gels_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gels_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gels_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'strideA': '150',
            'strideB': '150',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_laswp(self):
        out, err, exitcode = call_rocsolver_bench('-f laswp -n 10 --k1 15 --k2 20')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '20',
            'k1': '15',
            'k2': '20',
            'inc': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_larfg(self):
        out, err, exitcode = call_rocsolver_bench('-f larfg -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'inc': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_larf(self):
        out, err, exitcode = call_rocsolver_bench('-f larf -m 10 --side L')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'side': 'L',
            'm': '10',
            'n': '10',
            'inc': '1',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_larft(self):
        out, err, exitcode = call_rocsolver_bench('-f larft -n 10 --storev C -k 5')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'direct': 'F',
            'storev': 'C',
            'n': '10',
            'k': '5',
            'ldv': '10',
            'ldt': '5',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_larfb(self):
        out, err, exitcode = call_rocsolver_bench('-f larfb -n 10 -m 15 --direct F --side L --storev C -k 5')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'side': 'L',
            'trans': 'N',
            'direct': 'F',
            'storev': 'C',
            'm': '15',
            'n': '10',
            'k': '5',
            'ldv': '15',
            'ldt': '5',
            'lda': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_latrd(self):
        out, err, exitcode = call_rocsolver_bench('-f latrd -n 10 -k 5')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'k': '5',
            'lda': '10',
            'ldw': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_labrd(self):
        out, err, exitcode = call_rocsolver_bench('-f labrd -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'nb': '10',
            'lda': '15',
            'ldx': '15',
            'ldy': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_bdsqr(self):
        out, err, exitcode = call_rocsolver_bench('-f bdsqr --uplo U -n 15 --nu 10 --nv 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '15',
            'nv': '10',
            'nu': '10',
            'nc': '0',
            'ldv': '15',
            'ldu': '10',
            'ldc': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_steqr(self):
        out, err, exitcode = call_rocsolver_bench('-f steqr -n 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'evect': 'N',
            'n': '15',
            'ldc': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_stedc(self):
        out, err, exitcode = call_rocsolver_bench('-f stedc -n 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'evect': 'N',
            'n': '15',
            'ldc': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_stein(self):
        out, err, exitcode = call_rocsolver_bench('-f stein -n 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '15',
            'nev': '5',
            'ldz': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_lasyf(self):
        out, err, exitcode = call_rocsolver_bench('-f lasyf -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nb': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potf2(self):
        out, err, exitcode = call_rocsolver_bench('-f potf2 -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potf2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potf2_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potf2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potf2_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrf(self):
        out, err, exitcode = call_rocsolver_bench('-f potrf -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potrf_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potrf_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrs(self):
        out, err, exitcode = call_rocsolver_bench('-f potrs -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrs_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potrs_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potrs_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potrs_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideB': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_posv(self):
        out, err, exitcode = call_rocsolver_bench('-f posv -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_posv_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f posv_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_posv_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f posv_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideB': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potri(self):
        out, err, exitcode = call_rocsolver_bench('-f potri -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potri_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potri_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_potri_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f potri_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2_npvt(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2_npvt -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2_npvt_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2_npvt_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2_npvt_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2_npvt_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf_npvt(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf_npvt -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf_npvt_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf_npvt_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf_npvt_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf_npvt_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrf_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2 -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getf2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getf2_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqr2(self):
        out, err, exitcode = call_rocsolver_bench('-f geqr2 -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqr2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqr2_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqr2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqr2_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqrf(self):
        out, err, exitcode = call_rocsolver_bench('-f geqrf -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqrf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqrf_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqrf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqrf_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqrf_ptr_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqrf_ptr_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerq2(self):
        out, err, exitcode = call_rocsolver_bench('-f gerq2 -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerq2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gerq2_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerq2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gerq2_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerqf(self):
        out, err, exitcode = call_rocsolver_bench('-f gerqf -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerqf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gerqf_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gerqf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gerqf_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geql2(self):
        out, err, exitcode = call_rocsolver_bench('-f geql2 -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geql2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geql2_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geql2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geql2_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqlf(self):
        out, err, exitcode = call_rocsolver_bench('-f geqlf -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqlf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqlf_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_geqlf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f geqlf_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelq2(self):
        out, err, exitcode = call_rocsolver_bench('-f gelq2 -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelq2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gelq2_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelq2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gelq2_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelqf(self):
        out, err, exitcode = call_rocsolver_bench('-f gelqf -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelqf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gelqf_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gelqf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gelqf_strided_batched -m 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '10',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrs(self):
        out, err, exitcode = call_rocsolver_bench('-f getrs -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrs_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrs_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getrs_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getrs_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideP': '10',
            'strideB': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gesv(self):
        out, err, exitcode = call_rocsolver_bench('-f gesv -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gesv_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gesv_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gesv_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gesv_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'nrhs': '10',
            'lda': '10',
            'ldb': '10',
            'strideA': '100',
            'strideP': '10',
            'strideB': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gesvd(self):
        out, err, exitcode = call_rocsolver_bench('-f gesvd -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'ldu': '15',
            'ldv': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)
 
    def test_gesvd_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gesvd_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideS': '10',
            'ldu': '15',
            'strideU': '225',
            'ldv': '10',
            'strideV': '100',
            'strideE': '9',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gesvd_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gesvd_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'left_svect': 'N',
            'right_svect': 'N',
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideS': '10',
            'ldu': '15',
            'strideU': '225',
            'ldv': '10',
            'strideV': '100',
            'strideE': '9',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_trtri(self):
        out, err, exitcode = call_rocsolver_bench('-f trtri -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_trtri_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f trtri_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_trtri_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f trtri_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'diag': 'N',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri(self):
        out, err, exitcode = call_rocsolver_bench('-f getri -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_outofplace(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_outofplace -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'ldc': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_outofplace_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_outofplace_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'ldc': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_outofplace_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_outofplace_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'ldc': '10',
            'strideC': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt_outofplace(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt_outofplace -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'ldc': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt_outofplace_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt_outofplace_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'ldc': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_getri_npvt_outofplace_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f getri_npvt_outofplace_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'ldc': '10',
            'strideC': '100',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gels(self):
        out, err, exitcode = call_rocsolver_bench('-f gels -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gels_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gels_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gels_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gels_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'trans': 'N',
            'm': '15',
            'n': '10',
            'nrhs': '10',
            'lda': '15',
            'ldb': '15',
            'strideA': '150',
            'strideB': '150',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebd2(self):
        out, err, exitcode = call_rocsolver_bench('-f gebd2 -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebd2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gebd2_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebd2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gebd2_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebrd(self):
        out, err, exitcode = call_rocsolver_bench('-f gebrd -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebrd_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gebrd_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_gebrd_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f gebrd_strided_batched -n 10 -m 15')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'm': '15',
            'n': '10',
            'lda': '15',
            'strideA': '150',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytf2(self):
        out, err, exitcode = call_rocsolver_bench('-f sytf2 -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytf2_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f sytf2_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytf2_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f sytf2_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytrf(self):
        out, err, exitcode = call_rocsolver_bench('-f sytrf -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytrf_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f sytrf_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

    def test_sytrf_strided_batched(self):
        out, err, exitcode = call_rocsolver_bench('-f sytrf_strided_batched -n 10')
        self.assertEqual(err, '')
        self.assertEqual(exitcode, 0)

        args = self.parse_arguments(out)
        expected_args = {
            'uplo': 'U',
            'n': '10',
            'lda': '10',
            'strideA': '100',
            'strideP': '10',
            'batch_c': '1',
        }
        self.assertEqual(args, expected_args)

        results = self.parse_results(out)
        self.assertGreaterEqual(float(results['cpu_time']), 0)
        self.assertGreaterEqual(float(results['gpu_time']), 0)

if __name__ == '__main__':
    unittest.main()
