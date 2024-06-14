#!/bin/bash
bitwidth=$1
echo "Running Training"
if [ -z "$1" ]
  then
    echo "No bitwidth supplied training float"
  else
    echo "bitwidth = \"$1\""
fi

data="dataset=fhswf/TinyStoriesV2_cleaned dataset.dataloader.batch_size=32"
run="run=train run.epochs=1"
Omodels=( BENCH_gpt2_ReBNP_small BENCH_gpt2_ReBNP_smaller BENCH_gpt2_ReBNP_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReLN_small BENCH_gpt2_ReLN_smaller BENCH_gpt2_ReLN_tiny )
models=( BENCH_gpt2_ReBNP_nano BENCH_gpt2_ReBNT_nano BENCH_gpt2_ReLN_nano )

for model in ${models[@]}
do
    echo qtransform $run model=$model $data
    qtransform $run model=$model $data
done

#!/bin/bash
bitwidth=$1
echo "Running Training"
if [ -z "$1" ]
  then
    echo "No bitwidth supplied training float"
  else
    echo "bitwidth = \"$1\""
fi


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