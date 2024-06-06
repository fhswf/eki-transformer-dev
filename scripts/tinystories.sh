data="dataset=lhf_tinystories dataset.dataloader.batch_size=32"
run="run=train run.epochs=1 "
Omodels=( BENCH_gpt2_ReBNP_small BENCH_gpt2_ReBNP_smaller BENCH_gpt2_ReBNP_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReLN_small BENCH_gpt2_ReLN_smaller BENCH_gpt2_ReLN_tiny )
models=( BENCH_gpt2_ReBNP_nano BENCH_gpt2_ReBNT_nano BENCH_gpt2_ReLN_nano )

for model in ${models[@]}
do
    echo qtransform $run model=$model $data
    qtransform $run model=$model $data
done