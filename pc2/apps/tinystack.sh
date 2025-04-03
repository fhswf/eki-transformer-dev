#!/bin/bash
# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

# Setup the python environment for model training and evaluation and export

# check if this script is running in slurm

if [ -z "$SLURM_JOB_ID" ]; then
  echo "Vanilla Node execution"
else
  echo "This script is running inside a SLURM job with SLURM_JOB_ID=$SLURM_JOB_ID"
  # if system is pc2: 
  if [ -z "$PC2SYSNAME" ]; then
    echo "Slurm execution on somthing other then pc2 not supported yet"
  else
    # load modules
    module load lang/Python/3.11.5-GCCcore-13.2.0
    module load tools/git/2.41.0-GCCcore-12.3.0-nodocs
    module load lang/Tkinter/3.11.5-GCCcore-13.2.0
    # set env to be compatible with pc2
    export WORK_HOME="$PC2PFS/hpc-prf-ekiapp/paulw"
    export PYTHONUSERBASE="$SB_PRF_HOME/.local"
    mkdir -p $PYTHONUSERBASE
    export HF_HOME="$PC2PFS/hpc-prf-ekiapp/paulw/hf_cache"
    export TRANSFORMERS_CACHE=$HF_HOME
    mkdir -p $HF_HOME
    # does not work atm:
    #export OMP_NUM_THREADS=1
    #export ORT_SINGLE_THREAD=1
  fi
fi

python3 -m venv $WORK_HOME/venv/
source $WORK_HOME/venv/bin/activate
# cd $SBHOME/git && git clone https://github.com/fhswf/eki-transformer-dev

pip install --upgrade pip
pip install git+https://github.com/fhswf/eki-transformer-dev.git@tiny-stack#subdirectory=qtransform
#pip install -e $HOME/git/eki-transformer-dev/qtransform

pip list


if [ -z "$QTRANSFORM_RUN_XYZ" ]; then
  echo "QTRANSFORM_RUN_XYZ not set"
else
  echo "QTRANSFORM_RUN_XYZ set"
fi


data="dataset=tinystack dataset.dataloader.batch_size=32 dataset.root_path=$WORK_HOME/.qtransform/datasets tokenizer=tinystack"
run="run=train run.epochs=1 model=tinystack run.export=True debug=True +trace=True"
quant="quantization=qat quantization/model=SLURM_BENCH"
# models=( BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller )
models=( tinystack )
for model in ${models[@]}
do
    echo qtransform $run model=$model $data $quant +run.max_iters=50
    qtransform $run model=$model $data $quant +run.max_iters=50
done