#!/bin/python3
import subprocess
import os
from itertools import product
import pathlib

# one epoch, depending on dataset size and batch size. about 35393 batches for size 32
# we run for 2000 iters as sampling
static_args = "run=train run.epochs=1 +model.type=CHECKPOINT \
debug=True \
wandb.init.project=qtransform-energybench run.export=True +run.max_iters=100 +trace=True\
" 
#  # +trace=True \
# run.export=True +run.max_iters=5000 \

# done in slurm specific batch script
# work_env = ["dataset.root_path=$WORK_HOME/.qtransform/datasets"]

datasets = [
    "dataset=tsV2 tokenizer=hf tokenizer.encoding=fhswf/BPE_GPT2_TinyStoriesV2_cleaned_2048"
]

hyperparam_combinations =  [
    "dataset.dataloader.batch_size=32", 
    # "dataset.dataloader.batch_size=8"
]

models = [
    "model=NEW_BENCH2",    
]
model_config = [
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=1 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=2 model.args.n_embd=512 model.args.block_size=512",
    
    # "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=256",
    "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=256",
    #"model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=256",
    #"model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=256",
    #"model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=256",
    #"model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=256",
    #"model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=128 model.args.block_size=512",
    #"model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=512",
    #"model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=512",
    #"model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=256 model.args.block_size=512",
    #"model.args.n_layer=1 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=512",
    #"model.args.n_layer=2 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=512",
    #"model.args.n_layer=3 model.args.n_head=4 model.args.n_embd=512 model.args.block_size=512",
    
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=128",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=256",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=128 model.args.block_size=512",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=256 model.args.block_size=512",
    # "model.args.n_layer=1 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=512",
    # "model.args.n_layer=2 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=512",
    # "model.args.n_layer=3 model.args.n_head=6 model.args.n_embd=512 model.args.block_size=512",
]

model_args_dropout=[
    "model.args.dropout=0.0", 
    #"model.args.dropout=0.1", 
    #"model.args.dropout=0.2"
]

model_args_norm_layer=[
    #"model.args.norm_layer=None",
    "model.args.norm_layer=BatchNormTranspose", 
    #"model.args.norm_layer=LayerNorm", 
    #"model.args.norm_layer=BatchNormIdPure"
]

model_args_pos_layer=[
    "model.args.pos_layer=learned"
]

optim = [
    #"optim.args.weight_decay=0.0",
    #"optim.args.weight_decay=0.1",
    "optim.args.weight_decay=0.2",    
]

quant = [
    #"quantization=qat quantization/model=SLURM_BENCH2",
    # "quantization=qat quantization/model=SLURM_BENCH3",
    "quantization=qat quantization/model=FINN_PLUS_BENCH4",
    #"quantization=qat quantization/model=SLURM_BENCH5",
    #"quantization=qat quantization/model=SLURM_BENCH6",
    #"quantization=qat quantization/model=SLURM_BENCH7",
    #"quantization=qat quantization/model=SLURM_BENCH8"
    #"",
]

argument_combinations = list(product(datasets, hyperparam_combinations, models, model_config, model_args_dropout, model_args_pos_layer, model_args_norm_layer, optim, quant))
for args in argument_combinations:
    split_args = []
    for group in args:
        split_args.extend(group.split(" "))
    # split_args = list(args)
    call_args = \
        ["sbatch"] \
        + ["--export=WANDB_API_KEY="+os.environ["wandbkey"]] \
        + [ os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps/generic_run.sh")] \
        + split_args \
        + static_args.split(" ")
        #+ work_env
    print(f"Running call: {' '.join(call_args)}")    
    # result = subprocess.run(["which", "sbatch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = subprocess.run(call_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("stdout: " + str(result.stdout))
    print("stderr: " + str(result.stderr))
