#!/bin/bash
#Script tests the functionality of named pipes for transfering checkpoint and onnx filepaths 
#onto multiple processes
#In this case, a tiny gpt2 model is trained with shakespeare and the last checkpoint is used
#for inference

#!!!
#   TODO: move qtransform module into site packages
#!!!

PIPE_NAME="/tmp/qtransform.fifo"
if ! [[ -p ${PIPE_NAME} ]]; then
    mkfifo ${PIPE_NAME}
fi

#add signal handler to kill current background processes
#from https://stackoverflow.com/a/2173421
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

python -m ../qtransform run=train \
    model=gpt_2_h2l2e256b64_ReBN \
    dataset=huggingface dataset.name=tiny_shakespeare +export=True \
    dataset/tokenizer=tiktoken dataset.tokenizer.encoding=gpt2 \
    run.epochs=1 run.max_iters=10 run.export=False \
    debug=True \
    pipe=${PIPE_NAME} 2>&1 >/dev/null & #in background to retrieve data from pipe
#consume path name
checkpoint=$(cat ${PIPE_NAME})
if [[ -z ${checkpoint} ]]; then
    echo -e "checkpoint not set by previous process"
    exit 1
fi

#run inference, no need to pipe into fifo
python -m ../qtransform run=infer run.from_checkpoint=${checkpoint}
rm -f ${PIPE_NAME}
exit 0