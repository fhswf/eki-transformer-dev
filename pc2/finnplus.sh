
#!/bin/bash

# mounted storage is a bit too slow for compliation, hence cpy everything to ramdisk
# ram disk: /dev/shm/

# Setup the python environment for model training and evaluation and export
module load lang/Python/3.10.8-GCCcore-12.2.0-bare
python3.10 -m venv /dev/shm/env/
source /dev/shm/env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup the FPGA development environment
# not sure which versions are necessary, this one is the same as the radioml 
module load fpga
module load xilinx/xrt/2.14
module load xilinx/vitis/22.2

# Prepare for running FINN in Singularity containers
module load system singularity
export SINGULARITY_CACHEDIR=/dev/shm/singularity-cache/
export SINGULARITY_TMPDIR=/dev/shm/singularity-tmp/
# prebuild finn dev container under common project dir
export FINN_SINGULARITY=$PC2DATA/hpc-prf-ekiapp/FINN_IMAGES/xilinx/finn_dev.sif

# Prepare FINN to find the Vitis/Vivado installation
export FINN_XILINX_PATH=/opt/software/FPGA/Xilinx/
export FINN_XILINX_VERSION=2022.2

# Somehow these options are required to get FINN running on the cluster...
export LC_ALL="C"
export PYTHONUNBUFFERED=1
export XILINX_LOCAL_USER_DATA="no"

# If a path to a FINN installation is specified, move it to some faster storage
# location
if [[ -d "$FINN" ]]; then
  # Copy FINN to the ramdisk
  cp -r "$FINN" /dev/shm/finn/
  # Redirect the path specified via environment variable to use the copy
  export FINN=/dev/shm/finn/
fi;

# Generate FINN build outputs and temporaries to the ramdisk
export FINN_HOST_BUILD_DIR=/dev/shm/finn-build

# Write the command line to be executed to the log
echo "$@"
# Forward all command line arguments as the command line to be run as the job
eval "$@"

# If FINN actually produced build outputs
if [[ -d "$FINN_HOST_BUILD_DIR" && $DEBUG ]]; then
  # Generate a (hopefully) unique name for debugging output
  DEBUG_OUTPUT="fh-swf-build-$(hostname)-$(date +'%Y-%m-%d-%H-%M-%S').tar.gz"
  # For debugging purposes collect all build outputs from the ramdisk
  tar -zcf "$DEBUG_OUTPUT" /dev/shm/finn-build
fi;


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