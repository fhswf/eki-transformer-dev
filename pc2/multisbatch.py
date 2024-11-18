#!/bin/python3
import subprocess
import os
from itertools import product
import pathlib

static_args = ["run=train run.epochs=1 +model.type=CHECKPOINT quantization=qat run.export=True debug=True +trace=True +run.max_iters=50000"]

# done in slurm specific batch script
# work_env = ["dataset.root_path=$WORK_HOME/.qtransform/datasets"]

datasets = [
    "dataset=tsV2 tokenizer=hf tokenizer.encoding=fhswf/BPE_GPT2_TinyStoriesV2_cleaned_2048"
]

hyperparam_combinations =  [
    "dataset.dataloader.batch_size=32"
]

models = [
    "model=NEW_BENCH2",    
]

quant = [
    "quantization=qat quantization/model=SLURM_BENCH",
]

argument_combinations = list(product(datasets, hyperparam_combinations, models, quant))
for args in argument_combinations:
    split_args = []
    for group in args:
        split_args.extend(group.split())

    call_args = \
        ["sbatch"] \
        + ["--export=WANDB_API_KEY="+os.environ["wandbkey"]] \
        + [ os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps/generic_run.sh")] \
        + split_args \
        + static_args \
        + work_env
    result = subprocess.run(call_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("call: " + str(call_args))
    print("stdout: " + str(result.stdout))
    print("stderr: " + str(result.stderr))
