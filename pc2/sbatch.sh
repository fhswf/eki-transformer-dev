#!/bin/bash
#SBATCH -t 1:0:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem 64G
#SBATCH -J "qtransformtest"
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

eval apps/tinyStories.sh