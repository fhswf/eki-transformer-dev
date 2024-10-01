#!/bin/bash
usage () {
    cat <<HELP_USAGE

    $0  [-a] -f <file>

   -a  All the instances.
   -f  File to write all the log lines
HELP_USAGE
}

bitwidth=$1

echo "Running Training"
if [ -z "$1" ]
  then
    echo "No bitwidth supplied training float"
  else
    echo "bitwidth = \"$1\""
fi



function parallel {
  local time1=$(date +"%H:%M:%S")
  local time2=""

  # for the sake of the example, I'm using $2 as a description, you may be interested in other description
  echo "starting $2 ($time1)..."
  "$@" && time2=$(date +"%H:%M:%S") && echo "finishing $2 ($time1 -- $time2)..." &

  local my_pid=$$
  local children=$(ps -eo ppid | grep -w $my_pid | wc -w)
  children=$((children-1))
  if [[ $children -ge $max_children ]]; then
    wait -n
  fi
}



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