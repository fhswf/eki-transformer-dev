#!/bin/python3
import subprocess
import os
from itertools import product
import pathlib
import argparse

static_args = ""  

# Default models list (used if no CLI argument provided)
default_models = [
    "qonnx_NEW_BENCH2_tsV2_251105-15:52:58-defiant-feel__ep:1_1.onnx",    
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process FPGA models with sbatch')
parser.add_argument('--model', '-m', type=str, 
                    help='Model file to process (overwrites default models list)')
args = parser.parse_args()

# Use CLI model if provided, otherwise use default models
if args.model:
    models = [args.model]
else:
    models = default_models


models = [model for model in models]

argument_combinations = list(product(models))
for args in argument_combinations:
    split_args = []
    for group in args:
        split_args.extend(group.split(" "))
    # split_args = list(args)
    call_args = \
        ["sbatch"] \
        + [ os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps/finnplus2.sh")] \
        + split_args \
        + static_args.split(" ")
        #+ work_env
    print(f"Running call: {' '.join(call_args)}")    
    result = subprocess.run(call_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("stdout: " + str(result.stdout))
    print("stderr: " + str(result.stderr))
