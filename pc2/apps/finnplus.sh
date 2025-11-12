#!/bin/bash

#######################
### SLURM JOB CONFIG ##
#######################
#SBATCH -t 1:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q express
#SBATCH --cpus-per-task=16
#SBATCH --mem 128G

#SBATCH -J "qtransform-fpga"
# default is the normal partition, see Node Types and Partitions for PC2 options are "normal" "gpu" and  "dgx" "fpga"
#SBATCH -p normal
#SBATCH -A hpc-prf-ekiapp
#SBATCH --mail-type FAIL
#SBATCH --mail-user kuhmichel.max@fh-swf.de

# launch script with :
# sbatch name_of_this_file.sh

# monitor job with squeue or squeue_pretty or spredict
# cancle job with scancel JOBID or all jobs with scancel -u USERNAME

# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

#######################
# actual app/programm # -exclusive
#######################


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

    # Setup the FPGA development environment
    
    ml lang/Python/3.10.4-GCCcore-11.3.0
    ml devel/Autoconf/2.71-GCCcore-11.3.0
    ml lang/Bison/3.8.2-GCCcore-11.3.0
    ml lang/flex/2.6.4-GCCcore-11.3.0
    ml compiler/GCC/11.3.0
    ml lib/pybind11/2.9.2-GCCcore-11.3.0
    ml devel/Boost/1.79.0-GCC-11.3.0
    ml lib/fmt/9.1.0-GCCcore-11.3.0
    ml fpga xilinx/xrt/2.14
    ml lib/gurobi/1203
    module swap xilinx/u280 xilinx/u55c
    module load tools/git/2.41.0-GCCcore-12.3.0-nodocs
    # my home
    export MY_HOME="$HOME"
    export PFS_HOME="$PC2PFS/hpc-prf-ekiapp/maxkm"
    # set env to be compatible with pc2
    export DATASET_ROOT_PATH="$PC2PFS/hpc-prf-ekiapp/maxkm/.qtransform/datasets"
    echo "$PC2PFS"
    # use /dev/shm
    export RAMDISK=/dev/shm
    export WORK_HOME="$RAMDISK/work"
    mkdir -p $WORK_HOME
    export PYTHONUSERBASE="$WORK_HOME/.local"
    mkdir -p $PYTHONUSERBASE
    export HF_HOME="$PC2PFS/hpc-prf-ekiapp/hf_cache"
    mkdir -p $HF_HOME

    # does not work atm:
    #export OMP_NUM_THREADS=1
    #export ORT_SINGLE_THREAD=1
    # new finn plus python package
    python3 -m venv $WORK_HOME/venv/
    source $WORK_HOME/venv/bin/activate
  fi
fi

# install finn plus in python env
pip install --upgrade pip
# pip install finn-plus
pip install git+https://github.com/eki-project/finn-plus.git@transformer

# update finn plus
echo "Running finn plus update and config list"
finn deps update
# check install
# echo "Running finn test"
# finn test
# check current config
echo "Running finn config list"
finn config create $WORK_HOME
finn config list

ls -lash $WORK_HOME

# Write the command line to be executed to the log
echo "$@"

###################
# finn build process
###################

# FINN env variables to use ramdisk for build process and to find configs and stuff
mkdir -p $WORK_HOME/finn-build/
export FINN_HOST_BUILD_DIR=$WORK_HOME/finn-build
export FINN_BUILD_DIR=$WORK_HOME/finn-build
export FINN_DEPS=$WORK_HOME/finn-deps

# extract model name from onnx path
MODEL_PATH=$(echo $1 | rev | cut -d'/' -f1 | rev)
MODEL_NAME=$MODEL_PATH
echo "Model name extracted: $MODEL_NAME"

# copy onnx model to WORK_HOME
cp $1 $WORK_HOME/$MODEL_PATH
# copy input and output data to ramdisk ($1 + inp.npy and $1 + onnx_out.npy)
cp ${1}.inp.npy $WORK_HOME/
cp ${1}.onnx_out.npy $WORK_HOME/

# copy finn config and settings and build.yaml to ramdisk
cp -r  $MY_HOME/git/eki-transformer-dev/pc2/finn/* $WORK_HOME/

#change to work home
cd $WORK_HOME

ls -lash $WORK_HOME

# ## finnn build command with settings build config and models on the ramdisk
# finn run build 
echo "CALL_MODEL_NAME is set to $CALL_MODEL_NAME"
CALL_MODEL_NAME=$MODEL_NAME finn run build.py

ls -lash $WORK_HOME 
echo "$WORK_HOME/finn-build"
ls -lash $WORK_HOME/finn-build
echo "FINN build process finished $FINN_HOST_BUILD_DIR."
ls -lash $FINN_HOST_BUILD_DIR

# If FINN actually produced build outputs
if [[ -d "$FINN_HOST_BUILD_DIR" ]]; then
  # use model name and timestamp for output tarball
  OUTPUT_NAME="${MODEL_NAME}_finn_build_outputs_$(date +%Y%m%d_%H%M%S).tar.gz"
  echo "Collecting finn build outputs into $OUTPUT_NAME"
  # For debugging purposes collect all build outputs from the ramdisk
  tar -zcvf "$OUTPUT_NAME" "$FINN_HOST_BUILD_DIR"
  mkdir -p $MY_HOME/finn-build-outputs/
  cp "$OUTPUT_NAME" $MY_HOME/finn-build-outputs/
fi;
