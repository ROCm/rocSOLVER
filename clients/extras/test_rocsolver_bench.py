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
            'ldb': '15'
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

if __name__ == '__main__':
    unittest.main()
