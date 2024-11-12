#!/bin/bash
# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

# Setup the python environment for model training and evaluation and export
	
module load lang/Python/3.11.5-GCCcore-13.2.0
module load tools/git/2.41.0-GCCcore-12.3.0-nodocs
module load lang/Tkinter/3.11.5-GCCcore-13.2.0

export SB_PRF_HOME="$PC2PFS/hpc-prf-ekiapp/maxkm"
export PYTHONUSERBASE="$SB_PRF_HOME/.local"
mkdir -p $PYTHONUSERBASE

python3 -m venv $SB_PRF_HOME/venv/
source $SB_PRF_HOME/venv/bin/activate
# cd $SBHOME/git && git clone https://github.com/fhswf/eki-transformer-dev

pip install --upgrade pip
# pip install git+https://github.com/fhswf/eki-transformer-dev.git@main#subdirectory=qtransform 
pip install -e $HOME/git/eki-transformer-dev/qtransform

data="dataset=tsV2 dataset.dataloader.batch_size=32 dataset.root_path=$SB_PRF_HOME/.qtransform/datasets tokenizer=hf tokenizer.encoding=fhswf/BPE_GPT2_TinyStoriesV2_cleaned_2048"
run="run=train run.epochs=1 +model.type=CHECKPOINT  run.export=True +trace=True"
quant="quantization=qat quantization/model=NEW_BENCH2b"
models=( BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller )
# models=( BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller )
models=( NEW_BENCH2 ) 
for model in ${models[@]}
do
    echo qtransform $run model=$model $data $quant +run.max_iters=500
    HF_HOME="$PC2PFS/hpc-prf-ekiapp/hf_cache" qtransform $run model=$model $data $quant +run.max_iters=500
done