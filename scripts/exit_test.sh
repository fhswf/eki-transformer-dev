#!/bin/bash

# exits if no nvidia is available. Return the most free device and its util in percent 
get_free_device() {
  local active_devices=`echo $CUDA_VISIBLE_DEVICES | tr "," " "`
  local lowest_device=99
  local lowest_util=1.0
  for device_id in ($active_devices)
  do
    local util_active_device=$(nvidia-smi --query-gpu="utilization.memory" --format="csv,noheader,nounits" --id=$device_id)
    if [ $util_active_device < $lowest_util]; then
      local lowest_device=$device_id
      local lowest_util=$util_active_device
    fi
  done
  if [$lowest_device != 99]; then
    echo "$lowest_device,$lowest_util"
  else
    echo "No Cuda device found"
    return 1
  fi
}

check_status_return() {
}

# cmdXYZ:err_counter
local commands=( "python3 -c 'import torch; print(torch.__version__)':0"
                 "python3 -c 'import torch; print(torch.__version__)':0"
               )
local max_children=0
local max_errs=5

# some vars for the loop logic
local pid_status
local pid
local pids=()

# while commands not empty
while [${#commands[@]} -gt 0]; do
    # extract command and error_counter:
    ce=${commands[0]}
    command=${ce%%:*}
    err_counter=${ce#*:}
    # remove first element of the queue:
    commands=("${commands[@]:1}")
    
    # remove job from queue if err_counter > max_errs

    # check if max process is reached 
    local my_pid=$$
    local children=$(ps -o ppid= | grep -w $my_pid | wc -l)
    if [[ $children -eq $max_children ]]; then
        wait -n  # Wait for any child process to finish
        pid_status=$? # exit code of finished job
        # TODO caputure potential hangup. if pid is still alive let the sricpt continue
        # TODO if execution failed and exit code was greater than 1 increase error counter and appand command back to command queue
    fi
    # start a new job
    dev_util=$(get_free_device)
    dev_id=$(echo $dev_util | awk  -F, '{print $1}')
    dev_util=$(echo $dev_util | awk -F, '{print $2}')
    $command & # TODO pipe std out to dev null and set CUDA visible device to dev_id
    pid=$! # capture pid
    sleep 180 # wait 3min for potential hangup

    # TODO caputure potential hangup. if pid is still alive let the sricpt continue
    # TODO if execution failed and exit code was greater than 2 increase error counter and appand command back to command queue
    # TODO dont increase error counter if exit code was 2 !
    
    # next job can be launched 
    # continue loop

    if [${#commands[@]} -eq 0]; then
        # wait for all jobs to finish with a loop 
        # so like: get all childs 
        # wait -n
        # if failed try again 
        # when all jobs are done exit loop
        while :; do
            children=$(ps -o ppid= | grep -w $my_pid | wc -l)
            if [[ $children -gt $0 ]]; then
                wait -n  # Wait for any child process to finish
                pid_status=$? # exit code of finished job
                # TODO caputure potential hangup. if pid is still alive let the sricpt continue
                # TODO if execution failed and exit code was greater than 1 increase error counter and appand command back to command queue
                
                # continue outer loop on errors
                if [ $pid_status -nt 0]; then
                    break
                fi
            else
                # all jobs are done, exit script
                # TODO give overview of succeeded and failed jobs
                echo "all jobs done"
                return 0
            fi
        done
    fi

done























# Eine Funktion, die einen Job simuliert
run_job() {
  sleep "$1"
  echo "Fertig mit Schlafzeit $1"
  return "$2"  # Der gewünschte Exit-Code
}

# Starte Jobs im Hintergrund und speichere ihre PIDs
run_job 2 0 &
pid1=$!
run_job 12 1 &
pid2=$!
run_job 2 0 &
pid3=$!

# Warte auf alle Jobs und speichere ihre Exit-Codes
wait "$pid1"
exit_code1=$?
echo "1"
wait "$pid2"
exit_code2=$?
echo "2"
wait "$pid3"
exit_code3=$?
echo "3"

# Ausgabe der Exit-Codes
echo "Exit-Code von PID $pid1: $exit_code1"
echo "Exit-Code von PID $pid2: $exit_code2"
echo "Exit-Code von PID $pid3: $exit_code3"