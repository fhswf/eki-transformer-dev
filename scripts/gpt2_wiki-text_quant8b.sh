data="dataset=huggingface dataset.name=wikitext dataset.subset=wikitext-103-raw-v1"
tokenizer="dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2"
models=( BENCH_gpt2_ReBNP_smaller BENCH_gpt2_ReBNP_small BENCH_gpt2_ReBNP_gpt2small )
quant_models=( BENCH_8b_gpt2_2 BENCH_8b_gpt2_3 )

for model in ${models[@]}
do
    for quant in  ${quant_models[@]}
    do
        qtransform run=train model=$model run.epochs=2 run.max_iters=5000 $data $tokenizer +export=True quantization=qat quantization/model=$quant
    done
done