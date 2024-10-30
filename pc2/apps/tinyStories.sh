#!/bin/bash
# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

# Setup the python environment for model training and evaluation and export
# module load lang/Python/3.10.8-GCCcore-12.2.0-bare    
module load lang/Python/3.11.3-GCCcore-12.3.0 
module load tools/git/2.41.0-GCCcore-12.3.0-nodocs

export SB_PRF_HOME="$PC2PFS/hpc-prf-ekiapp/maxkm"
export PYTHONUSERBASE="$SB_PRF_HOME/.local"
makedir -p $PYTHONUSERBASE

python3.11 -m venv $SB_PRF_HOME/venv/
source $SB_PRF_HOME/venv/bin/activate
# cd $SBHOME/git && git clone https://github.com/fhswf/eki-transformer-dev

pip install --upgrade pip
# pip install git+https://github.com/fhswf/eki-transformer-dev.git@main#subdirectory=qtransform 
pip install -e $HOME/git/eki-transformer-dev

data="dataset=lhf_tinystories dataset.dataloader.batch_size=32 dataset.root_path=$SB_PRF_HOME/.qtransform/datasets"
run="run=train run.epochs=1 "
quant="quantization=qat quantization/model=BENCH_4b_gpt2"
models=( BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller )

for model in ${models[@]}
do
    echo qtransform $run model=$model $data $quant +run.max_iters=500
    HF_HOME="$PC2PFS/hpc-prf-ekiapp/hf_cache" qtransform $run model=$model $data $quant +run.max_iters=500
done