
#!/bin/bash

#######################
### SLURM JOB CONFIG ##
#######################
#SBATCH -t 1:0:0  !!TODO!!
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16 !!TODO!!
#SBATCH --mem 64G
#SBATCH -J "qtransform-bench"
# default is the normal partition, see Node Types and Partitions for PC2 options are "normal" "gpu" and  "dgx" "fpga"
#SBATCH -p fpga  ??TODO??
#SBATCH -A hpc-prf-ekiapp
#SBATCH --mail-type FAIL
#SBATCH --mail-user kuhmichel.max@fh-swf.de
#SBATCH --gres=gpu:a100:1  !!TODO!!

# launch script with :
# sbatch name_of_this_file.sh

# monitor job with squeue or squeue_pretty or spredict
# cancle job with scancel JOBID or all jobs with scancel -u USERNAME

# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

#######################
# actual app/programm #
#######################

# Setup the FPGA development environment
# taken from linus recommendations
ml lang/Python/3.10.4-GCCcore-11.3.0
ml devel/Autoconf/2.71-GCCcore-11.3.0
ml lang/Bison/3.8.2-GCCcore-11.3.0
ml lang/flex/2.6.4-GCCcore-11.3.0
ml compiler/GCC/11.3.0
ml lib/pybind11/2.9.2-GCCcore-11.3.0
ml devel/Boost/1.79.0-GCC-11.3.0
ml lib/fmt/9.1.0-GCCcore-11.3.0
ml fpga xilinx/xrt/2.14
module swap xilinx/u280 xilinx/u55c

# new finn plus pytohnm package
python3.10 -m venv /dev/shm/env/
source /dev/shm/env/bin/activate
pip install --upgrade pip
pip install finn-plus

# update finn plus
finn deps update
# check install
finn test
# check current config
finn config list

# Write the command line to be executed to the log
echo "$@"
# # Forward all command line arguments as the command line to be run as the job
# eval "$@"

# copy build.py to ramdisk
cp build.py /dev/shm/build.py
cd /dev/shm/

# copy models and other stuff to ramdisk
cp -r $PC2DATA/hpc-prf-ekiapp/FPGA_MODELS /dev/shm/FPGA_MODELS

export FINN_HOST_BUILD_DIR=/dev/shm/finn-build
## finnn build
# finn run build.py
finn run build.yaml /dev/shm/FPGA_MODELS/model.onnx --save-dir /dev/shm/finn-build --skip-onnx-checks

# If FINN actually produced build outputs
if [[ -d "$FINN_HOST_BUILD_DIR" && $DEBUG ]]; then
  # Generate a (hopefully) unique name for debugging output
  DEBUG_OUTPUT="fh-swf-build-$(hostname)-$(date +'%Y-%m-%d-%H-%M-%S').tar.gz"
  # For debugging purposes collect all build outputs from the ramdisk
  tar -zcf "$DEBUG_OUTPUT" /dev/shm/finn-build
fi;
