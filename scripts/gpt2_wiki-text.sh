data="dataset=huggingface dataset.name=wikitext dataset.subset=wikitext-103-raw-v1"
tokenizer="dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2"
models=( BENCH_gpt2_ReBNP_gpt2small2 )
quant_models=( BENCH_4b_gpt2_2 BENCH_4b_gpt2_3 BENCH_8b_gpt2_2 BENCH_8b_gpt2_3 )

for model in ${models[@]}
do
    qtransform run=train model=$model run.epochs=10 run.max_iters=500 $data $tokenizer dataset.dataloader.batch_size=32
done