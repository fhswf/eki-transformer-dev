#!/bin/bash

#######################
### SLURM JOB CONFIG ##
#######################
#SBATCH -t 4:0:0
#SBATCH -N 1
#SBATCH -n 1
# -q express
#SBATCH --cpus-per-task=16
#SBATCH --mem 128G
# --exclusive
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

    # n2
    module load lang/Python/3.10.4-GCCcore-11.3.0
    module load devel/Autoconf/2.71-GCCcore-11.3.0
    module load lang/Bison/3.8.2-GCCcore-11.3.0
    module load lang/flex/2.6.4-GCCcore-11.3.0
    module load lib/pybind11/2.9.2-GCCcore-11.3.0
    module load devel/Boost/1.79.0-GCC-11.3.0
    module load lib/fmt/9.1.0-GCCcore-11.3.0
    module load compiler/GCC/11.3.0
    # 2.14 22.2
    module load fpga
    module load xilinx/xrt/2.14
    # 2.16 24.2
    # module load xilinx/xrt/2.16 
    # module swap xilinx/vitis/24.2

    # Setup the FPGA development environment
    # otus
    # module load lang
    # module load lib
    # module load tools
    # module load devel
    # ml lang/Python/3.10.4-GCCcore-11.3.0
    # ml devel/ncurses/6.4
    # ml devel/Autoconf/2.71-GCCcore-11.3.0
    # ml lang/Bison/3.8.2-GCCcore-11.3.0
    # ml lang/flex/2.6.4-GCCcore-11.3.0
    # ml compiler/GCC/11.3.0
    # ml lib/pybind11/2.9.2-GCCcore-11.3.0
    # ml devel/Boost/1.79.0-GCC-11.3.0
    # ml lib/fmt/9.1.0-GCCcore-11.3.0
    # ml fpga xilinx/xrt/2.16
    # ml lib/gurobi/1203
    # module load xilinx/u55c
    # module load xilinx/vitis/22.2
    # module load xilinx/vivado/22.2


    # uni
    module load tools/git/2.41.0-GCCcore-12.3.0-nodocs

    ulimit -s unlimited

    # my home
    export MY_HOME="$HOME"
    export RAMDISK=/dev/shm
    export PFS_HOME="$PC2PFS/hpc-prf-ekiapp/maxkm"
    # set env to be compatible with pc2
    export DATASET_ROOT_PATH="$PC2PFS/hpc-prf-ekiapp/maxkm/.qtransform/datasets"
    pwd
    # use /dev/shm
    export WORK_HOME="$RAMDISK/work"
    #export HOME=$WORK_HOME
    mkdir -p $WORK_HOME
    export PYTHONUSERBASE="$WORK_HOME/.local"
    mkdir -p $PYTHONUSERBASE
    export HF_HOME="$PC2PFS/hpc-prf-ekiapp/hf_cache"
    mkdir -p $HF_HOME
    echo "$HF_HOME will be used for dataset storage"
    # does not work atm:
    #export OMP_NUM_THREADS=1
    #export ORT_SINGLE_THREAD=1
    # new finn plus python package
    python3 -m venv $WORK_HOME/venv/
    source $WORK_HOME/venv/bin/activate
  fi
fi


# also this helps with vitis: setenv XILINX_LOCAL_USER_DATA no 
# from https://github.com/Xilinx/finn/issues/1215
# and https://adaptivesupport.amd.com/s/article/63253?language=en_US
export XILINX_LOCAL_USER_DATA=no
# alternative set this to be the ramdisk XILINX_TCLSTORE_USERAREA
export XILINX_TCLSTORE_USERAREA=$WORK_HOME/xilinx_tclstore

# fix this error:
# WARNING: [Common 17-1221] Tcl app 'xsim' is out of date for this release. Please run tclapp::reset_tclstore and reinstall the app.
# ERROR: [Common 17-685] Unable to load Tcl app xilinx::xsim
# Caught error:
# ERROR: [Common 17-69] Command failed: ERROR: [Common 17-685] Unable to load Tcl app xilinx::xsim
# by running tclapp::reset_tclstore
# also run tclapp::update_support to update all tcl apps
echo "Resetting tclstore to avoid tcl app errors"
# create a temp tcl script to run the command
echo "tclapp::reset_tclstore" > $WORK_HOME/reset_tcl.tcl
echo "tclapp::update_support" >> $WORK_HOME/reset_tcl.tcl
# run the tcl script with vivado
vivado -mode tcl -source $WORK_HOME/reset_tcl.tcl
# remove the temp tcl script
rm $WORK_HOME/reset_tcl.tcl

#
# FINN env variables to use ramdisk for build process and to find configs and stuff
mkdir -p $WORK_HOME/finn-build/
export FINN_HOST_BUILD_DIR=$WORK_HOME/finn-build
export FINN_BUILD_DIR=$WORK_HOME/finn-build
export FINN_DEPS=$WORK_HOME/finn-deps

# install finn plus in python env
pip install --upgrade pip
# pip install finn-plus
pip install git+https://github.com/DeepCowProductions/finn-plus.git@transformer-maxdev

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

# finn plus needs wget binray mak sure its there
which wget || (echo "wget not found, please install wget" && exit 1)

# create a sym link ncurses 5 to newer version to avoid errors in vitis, links go to writeable user lib dir and come from module devel/ncurses/6.4
# vitis needs libtinfo.so.5 from ncurses we get a newser version
# https://wiki.archlinux.org/title/Xilinx_Vivado?useskin=vector
# get current libtinfo version
export LD_LIBRARY_PATH=/opt/software/FPGA/Xilinx/Vivado/2024.2/lib/lnx64.o/:$LD_LIBRARY_PATH
TINFO_LIB=$(ls /usr/lib*/libtinfo.so.* | grep -E 'libtinfo.so.[0-9]+$' | head -n 1)
echo "Using tinfo lib: $TINFO_LIB"
# create symlink in user local lib dir
mkdir -p $HOME/.local/lib/
ln -s $TINFO_LIB $HOME/.local/lib/libtinfo.so.5
# make sure local lib path is in ld library path
export LD_LIBRARY_PATH=$HOME/.local/lib/:$LD_LIBRARY_PATH





# Write the command line to be executed to the log
echo "$@"

###################
# finn build process
###################

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

# call addtional streamline 
# TODO change paths names in passes.yaml
# sed ...

# # copy passes.yaml to ramdisk
# cp $MY_HOME/git/eki-transformer-dev/pc2/finn/passes.yaml $WORK_HOME/
# # copy adhoc_passes.py to ramdisk
# cp $MY_HOME/git/eki-transformer-dev/pc2/finn/adhoc_passes.py $WORK_HOME/
# echo "MODEL_NAME is set to $MODEL_NAME"
# onnx-passes -c passes.yaml -o streamlined.onnx $MODEL_NAME
# rm $MODEL_NAME
# mv streamlined.onnx $MODEL_NAME

# ## finnn build command with settings build config and models on the ramdisk
# finn run build 
CALL_MODEL_NAME=$MODEL_NAME finn run build.py

ls -lash $WORK_HOME 
echo "$WORK_HOME/finn-build"
echo "FINN build process finished $FINN_HOST_BUILD_DIR."

# If FINN actually produced build outputs
if [[ -d "$FINN_HOST_BUILD_DIR" ]]; then
  # use model name and timestamp for output tarball
  OUTPUT_NAME="${MODEL_NAME}_finn_build_outputs_$(date +%Y%m%d_%H%M%S).tar.gz"
  echo "Collecting finn build outputs into $OUTPUT_NAME"
  # For debugging purposes collect all build outputs from the ramdisk
  tar -zcf "$OUTPUT_NAME" "$FINN_HOST_BUILD_DIR"
  mkdir -p $MY_HOME/finn-build-outputs/
  cp "$OUTPUT_NAME" $MY_HOME/finn-build-outputs/
fi;
