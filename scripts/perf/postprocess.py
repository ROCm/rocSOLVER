# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
# ########################################################################

import argparse
import pandas as pd

def append_units(label):
     if label.endswith('_time'):
         return label + '_us'
     else:
         return label

def operation_count(n, function):
    """Returns the number of operations required for a given function with an n-by-n matrix."""
    if function.startswith('getrf'):
        return (2 * n ** 3 / 3) - (n ** 2 / 2) - (n / 6)
    elif function.startswith('geqrf'):
        return (4 * n ** 3 / 3) + (2 * n ** 2) + (4 * n)
    elif function.startswith('getri'):
        return (4 * n ** 3 / 3) - (n ** 2) + (5 * n / 3)
    else: 
        raise ValueError(f'Could not calculate operation count for {function}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Processes the rocsolver-benchmark-suite results.')
    parser.add_argument('input_path',
            help='the input file name for the raw benchmark results')
    parser.add_argument('output_path',
            help='the output file name for the processed benchmark results')
    args = parser.parse_args()

    # load the data and filter on the function name
    df = pd.read_csv(args.input_path, encoding='utf-8')

    # rename columns for nicer labels
    df.rename(append_units, axis='columns', inplace=True)

    # prepend name column
    df.insert(0, 'name', df['precision'].str.cat(df['function']))

    # the op_count derived column calculation can't handle a mix of functions
    if not df['function'].empty and (df['function'] != df['function'][0]).all():
        raise ValueError('This script can only handle one function at a time.')
    function = df['function'][0]

    # append derived columns
    df['mean_gpu_time_us_per_matrix'] = df['gpu_time_us'].astype(float) / df.get('batch_c', 1)
    df['op_count'] = operation_count(df['n'], function).round().astype(int) * df.get('batch_c', 1)
    df['performance_gflops'] = df['op_count'] / df['gpu_time_us'] / 1000

    # save results
    df.to_csv(args.output_path, encoding='utf-8', index=False)
