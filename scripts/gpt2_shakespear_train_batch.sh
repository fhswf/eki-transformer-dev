data="dataset=huggingface dataset.name=tiny_shakespeare"
tokenizer="dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2"
models=( nanoGPT_shakespeare gpt_2_h4l4e256b64_ReBN gpt_2_h2l2e256b64_ReBN gpt_2_h4l4e256b64_ReLN  gpt_2_h2l2e256b64_ReLN )

for model in ${models[@]}
do
    qtransform run=train model=$model run.epochs=2 $data  dataset/tokenizer=tiktoken $tokenizer
done

for model in ${models[@]}
do
    qtransform run=train model=$model run.epochs=2 $data  dataset/tokenizer=tiktoken $tokenizer quantization=qat quantization/model=4b_default_gpt2_bn
done

data="dataset=huggingface dataset.name=openwebtext"
tokenizer="dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2"
models=( nanoGPT_shakespeare )

for model in ${models[@]}
do
    qtransform run=train model=$model run.epochs=4 $data  dataset/tokenizer=tiktoken $tokenizer
done

