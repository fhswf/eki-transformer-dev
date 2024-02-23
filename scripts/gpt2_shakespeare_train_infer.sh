#!/bin/bash
 #===============================================================================
 #
 # FILE: gpt2_shakespeare_train_infer.sh
 #
 # USAGE: ./gpt2_shakespeare_train_infer.sh
 #
 # DESCRIPTION: Trains a model, saves the checkpoint into a named pipe 
 #              and runs inference on the checkpoint.
 #              In this case, a tiny GPT2 model is trained with the tiny_shakespeare
 #              huggingface dataset with tiktoken GPT2 tokenization.
 #
 # OPTIONS: ---
 # REQUIREMENTS: python package qtransform (install with python -m setup install )
 # BUGS: ---
 # NOTES: ---
 # AUTHOR: Maik Botchkarev, botchkarev.maik@fh-swf.de
 # ORGANIZATION: Fachhochschule Südwestfalen, Iserlohn
 # REVISION: 07.02.2024
 #===============================================================================

pip show qtransform 2>&1 1>/dev/null
if [[ $? -ne 0 ]]; then
    echo -e "Python package qtransform not installed (python -m setup install within project directory)"
    exit 1
fi


PIPE_NAME="/tmp/qtransform.fifo"
if [[ ! -p ${PIPE_NAME} ]]; then
    mkfifo ${PIPE_NAME}
fi

#add signal handler to kill current background processes
#from https://stackoverflow.com/a/2173421
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

#train parameters
DATASET=huggingface
DATASET_NAME=tiny_shakespeare
TOKENIZER=tiktoken
ENCODING=gpt2
EPOCHS=1
MAX_ITERS=10
EVAL_EPOCH_INTERVAL=$(expr ${EPOCHS} / 10) #perform eval after specified amount of epochs
if [[ ${EVAL_EPOCH_INTERVAL} -eq 0 ]]; then
    EVAL_EPOCH_INTERVAL=${EPOCHS}
fi
EVAL_ITERS=$(expr ${MAX_ITERS} / 10) #perform eval after specified amount of epochs
if [[ ${EVAL_ITERS} -eq 0 ]]; then
    EVAL_ITERS=${MAX_ITERS}
fi
EXPORT=False

echo -e "\n---- Training model ----\n"

python -m qtransform run=train \
    model=gpt_2_h2l2e256b64_ReBN \
    dataset=${DATASET} dataset.name=${DATASET_NAME} \
    dataset/tokenizer=${TOKENIZER} dataset.tokenizer.encoding=${ENCODING} \
    run.epochs=${EPOCHS} run.max_iters=${MAX_ITERS} run.export=${EXPORT} \
    run.eval_epoch_interval=${EVAL_EPOCH_INTERVAL} run.eval_iters=${EVAL_ITERS} \
    debug=False \
    pipe=${PIPE_NAME} 2>&1 >/dev/null & #in background to retrieve data from pipe
#consume path name, might contain brackets
checkpoint=$(cat ${PIPE_NAME})
if [[ -z "${checkpoint}" ]]; then
    echo -e "checkpoint not set by previous process"
    exit 1
fi

#run inference, no need to pipe into fifo
NUM_SAMPLES=10 #generate num_samples 
MAX_NEW_TOKENS=500
TEMPERATURE=0.8
TOP_K=200
START="\n"
TO_FILE="output.txt"

python -m qtransform run=infer \
    run.from_checkpoint="${checkpoint}" \
    run.num_samples=${NUM_SAMPLES} \
    run.max_new_tokens=${MAX_NEW_TOKENS} \
    run.temperature=${TEMPERATURE} \
    run.top_k=${TOP_K} \
    run.start=${START} \
    run.to_file=${TO_FILE}
rm -f ${PIPE_NAME}
exit 0