#!/bin/bash

#######################
### SLURM JOB CONFIG ##
#######################
#SBATCH -t 0:30:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64G
#SBATCH -J "qtransform-bench"
# default is the normal partition, see Node Types and Partitions for PC2 options are "normal" "gpu" and  "dgx" "fpga"
#SBATCH -p gpu
#SBATCH -A hpc-prf-ekiapp
#SBATCH --mail-type FAIL
#SBATCH --mail-user kuhmichel.max@fh-swf.de
#SBATCH --gres=gpu:a100:1

# launch script with :
# sbatch name_of_this_file.sh

# monitor job with squeue or squeue_pretty or spredict
# cancle job with scancel JOBID or all jobs with scancel -u USERNAME

# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

##########################
### actual app/programm ##
##########################

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

    # ml devel/Autoconf/2.71-GCCcore-11.3.0
    # ml lang/Bison/3.8.2-GCCcore-11.3.0
    # ml lang/flex/2.6.4-GCCcore-11.3.0
    # ml compiler/GCC/11.3.0
    # ml lib/pybind11/2.9.2-GCCcore-11.3.0
    # ml devel/Boost/1.79.0-GCC-11.3.0
    # ml lib/fmt/9.1.0-GCCcore-11.3.0
    ml fpga   
    ml xilinx/xrt/2.14
    ml xilinx/vitis/24.2
    ml swap 
    ml xilinx/u280
    # set env to be compatible with pc2
    export DATASET_ROOT_PATH="$PC2PFS/hpc-prf-ekiapp/maxkm/.qtransform/datasets"

    # use /dev/shm
    export RAMDISK=/dev/shm
    export WORK_HOME="$RAMDISK/work"
    mkdir -p $WORK_HOME
    export PYTHONUSERBASE="$WORK_HOME/.local"
    export FINN_DEPS=$WORK_HOME/finn-deps
    mkdir -p $PYTHONUSERBASE
    export HF_HOME="$PC2PFS/hpc-prf-ekiapp/hf_cache"
    mkdir -p $HF_HOME
    # does not work atm:
    #export OMP_NUM_THREADS=1
    #export ORT_SINGLE_THREAD=1
  
    python3 -m venv $WORK_HOME/venv/
    source $WORK_HOME/venv/bin/activate
  fi
fi

# cd $SBHOME/git && git clone https://github.com/fhswf/eki-transformer-dev

pip install --upgrade pip

# pip install $WORK_HOME/git/eki-transformer-dev/qtransform
# # update repo 
# git -C $WORK_HOME/git/eki-transformer-dev/qtransform fetch 
# git -C $WORK_HOME/git/eki-transformer-dev/qtransform pull

pip install git+https://github.com/fhswf/eki-transformer-dev.git@main#subdirectory=qtransform 
finn deps update
# finn test
pip list

# maybe do this to run soe stuff via env? maybe better to do this in the actual app
# if [ -z "$QTRANSFORM_RUN_XYZ" ]; then
#   echo "QTRANSFORM_RUN_XYZ not set"
# else
#   echo "QTRANSFORM_RUN_XYZ set"
# fi

echo qtransform $* dataset.root_path=$DATASET_ROOT_PATH
qtransform $* dataset.root_path=$DATASET_ROOT_PATH