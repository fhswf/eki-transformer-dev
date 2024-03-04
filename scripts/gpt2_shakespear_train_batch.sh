data="dataset=huggingface dataset.name=tiny_shakespeare"
tokenizer="dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2"
models=( nanoGPT_shakespeare gpt_2_h4l4e256b64_ReBN gpt_2_h4l4e256b64_ReLN )

for model in ${models[@]}
do
    qtransform run=train model= run.epochs=2 $data  dataset/tokenizer=tiktoken $tokenizer
done

