data="dataset=lhf_tinystories dataset.dataloader.batch_size=32"
run="run=train run.epochs=1 "
quant="quantization=qat quantization/model=BENCH_4b_gpt2"
models=( BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_smaller  )

for model in ${models[@]}
do
    echo qtransform $run model=$model $data $quant +run.max_iters=30000
    qtransform $run model=$model $data $quant +run.max_iters=30000
done