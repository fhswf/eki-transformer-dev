data="dataset=lazyhuggingface dataset.name=wikitext dataset.subset=wikitext-2-raw-v1"
tokenizer="dataset/tokenizer=huggingface dataset.tokenizer.name=gpt2"
models=( BENCH_gpt2_ReBN_small BENCH_gpt2_ReBN_tiny BENCH_gpt2_ReBNP_small BENCH_gpt2_ReBNP_tiny BENCH_gpt2_ReBNT_small BENCH_gpt2_ReBNT_tiny BENCH_gpt2_ReLN_small BENCH_gpt2_ReLN_tiny )

quant_models=( BENCH_4b_gpt2_1 )

for model in ${models[@]}
do
    for quant in  ${quant_models[@]}
    do
        echo qtransform run=train model=$model run.epochs=2 $data $tokenizer +export=True quantization=qat quantization/model=$quant
        qtransform run=train model=$model run.epochs=2 $data $tokenizer +export=True quantization=qat quantization/model=$quant
    done
done