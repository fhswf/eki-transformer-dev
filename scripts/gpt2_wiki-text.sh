data="dataset=lazyhuggingface dataset.name=wikitext dataset.subset=wikitext-2-raw-v1"
tokenizer="dataset/tokenizer=huggingface dataset.tokenizer.name=gpt2"
models=( BENCH_gpt2_ReBN_small BENCH_gpt2_ReBN_tiny BENCH_gpt2_ReBNP_small BENCH_gpt2_ReBNP_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReLN_small BENCH_gpt2_ReLN_tiny )

for model in ${models[@]}
do
    echo qtransform run=train model=$model run.epochs=2 $data $tokenizer dataset.dataloader.batch_size=32 run.export=False
    qtransform run=train model=$model run.epochs=2 $data $tokenizer dataset.dataloader.batch_size=32 run.export=False
done