#!/bin/bash

# manadatory specify the time limit of your job days-hours:minutes days-hours:minutes:seconds hours:minutes:seconds
#SBATCH -t TIMELIMIT

# default is 1
#SBATCH -N NODES

# default is 1 A task is usually an MPI rank.
#SBATCH -n NTASKS

# run each task with NCPUS processors. 
#SBATCH --cpus-per-task=NCPUS

# default is memory-per-node/number-of-cores  memory per allocated cpu core, e.g. 1000M or 2G for 1000 MB or 2 GB respectively
#SBATCH --mem-per-cpu MEM

# memory per node
#SBATCH --mem MEM
	
# default is the file name of the job script NAME of the compute job
#SBATCH -J NAME

# default is the normal partition, see Node Types and Partitions for PC2 options are "normal" "gpu" and  "dgx" "fpga"
#SBATCH -p PARTITION

# not manadatory if you are only member of one compute-time project
#SBATCH -A PROJECT

# default is the default QoS of your project, Use the QoS QOS. For a description of QoS see
#SBATCH -q QOS
	
# default value is NONE, MAILTYPEcan be NONE, BEGIN, END, FAIL, REQUEUE, ALL.
#SBATCH --mail-type MAILTYPE
	
# your mail that should receive the mail notifications
#SBATCH --mail-user MAILADDRESS

# default value is 1, Kill the entire job if one task fails. Possible values are 0 or 1. 
#SBATCH --kill-on-bad-exit 
	
echo "Hello World"

# eg. (without the %)
#%!/bin/bash
#%SBATCH -t 2:00:00
#%SBATCH -N 2
#%SBATCH -n 10
#%SBATCH -J "great calculation"
#%SBATCH -p normal

#run your application here

# launch script with :
# sbatch name_of_this_file.sh

# monitor job with squeue or squeue_pretty or spredict
# cancle job with scancel JOBID or all jobs with scancel -u USERNAME


# for parralell jobs use srun inside this batchfile 
# eg.
#%SBATCH -t 2:00:00
#%SBATCH -n 8
#%SBATCH --cpus-per-task=4
#%SBATCH -J "great calculation"

#export OMP_NUM_THREADS=4
#srun ...

# export SLURM_CPU_BIND=cores,quiet and export OMPI_MCA_hwloc_base_report_bindings=false inside  ~/.bashrc. prevent srun meta output

# use gpus with #%SBATCH --gres=gpu:a100:3
# nocuta2 has a100 NVIDIA A100 40 GB with NVLINK 
# noctua1 has a40  NVIDIA A40 

# dgx system for testing on noctua2  with:  --qos=devel --partition=dgx --gres=gpu:a100:$NGPUs
# every gpu extra is minus 30 mion on max runtime