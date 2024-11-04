#!/bin/bash
print_usage () {
    cat <<HELP_USAGE

    $0  [-q (6b|5b|4b|3b)] [-p 1] [--max-proc 10]

   -q | --quant_bitwidth  if supplied uses qat and exports qonnx to local folder.
   -p | --parallel File   runs jobs on current CUDA Device in parrallel if applicable. Only uses one GPU per default
   --max-proc             maximum number of processes
HELP_USAGE

}POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -q|--quant_bitwidth)
      local BITWIDTH="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--parallel)
      local MAX_PARALLEL=$2    # just a flag with no values
      shift
      shift
      ;;
    --max-proc)
      local MAX_PROC=$2    # just a flag with no values
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1" 
      print_usage()
      exit 1
      ;;
    *)
      local POSITIONAL_ARGS+=("$1") # positional args without flags
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
echo "BITWIDTH     = ${BITWIDTH}"
echo "MAX_PARALLEL = ${MAX_PARALLEL}"

if [[ -n $1 ]]; then
    echo "Pos argument ignored:$1"
fi

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
}

# control vars 
local last_success=true
local err_counter=0
local num_processes=0
local pids=()
local pid
local status
# for all commands to be run 
for command in "${commands[@]}"; do
  # try and error loop for current process
  while true; do
     # get free gpu
    local dev_util=$(get_free_device)
    # try and launch process 
    # Überprüfen, ob die maximale Anzahl von Unterprogrammen erreicht ist
    if [ $num_processes -lt $MAX_PROC ]; then
        # Starten Sie ein neues Unterprogramm
        command & # TODO
        pid=$!
    fi
    sleep 120 # wait for 2 min and see weather the run launched
    status=$?
    pids+=($pid)
    ((num_processes++))

    
  done
done


  # wait for free gpu space
  while true; do
    # Überprüfen, ob die maximale Anzahl von Unterprogrammen erreicht ist
    if [ $num_processes -lt $MAX_PROC ]; then
        # Starten Sie ein neues Unterprogramm
        run_subprogram &
        pids+=($!)
        ((num_processes++))
    fi

    # remove pids from process list
    for pid in "${pids[@]}"; do
        if ! ps -p $pid > /dev/null; then
            pids=(${pids[@]/$pid})
            ((num_processes--))
        fi
    done
   
    if [$last_success]; then
      sleep 1
    else
      sleep 3600
    fi
  done


wait -n # wait for all sub proccesses to be done
echo "All jobs completed."












data="dataset=tsV2 tokenizer=TS2k"
run="run=train run.epochs=1 run.max_iters=20000 dataset.dataloader.batch_size=32 run.export=True"
model="model=gpt2"
model_cstr=( "MGPT-s512-t2048-l4-h8-e512-AReLU-NBatchNormTranspose-Plearned" )
qat="quantization=qat quantization/model=NEW_BENCH$bitwidth"

for cstr in ${model_cstr[@]}
do
    echo qtransform $run model=$model $data
    qtransform $run model=$model $data
done

python -m qtransform 









parallel() {
  local time1=$(date +"%H:%M:%S")
  local time2=""

  echo "starting $2 ($time1)..."
  "$@" && time2=$(date +"%H:%M:%S") && echo "finishing $2 ($time1 -- $time2)..." &

  local my_pid=$$
  local children=$(ps -eo ppid | grep -w $my_pid | wc -w)
  children=$((children-1))
  if [[ $children -ge $max_children ]]; then
    wait -n
  fi
}