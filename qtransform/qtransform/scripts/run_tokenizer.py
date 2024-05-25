from itertools import chain
import logging

from tokenizers import Tokenizer
log = logging.getLogger(__name__)

from transformers import AutoTokenizer
from datasets import load_dataset

import qtransform
args = [ "run=script", "run.file=none" ]
@qtransform.with_config(args, logging.INFO)
def run_standalone(cfg):
    log.info(f"running {__file__}")
    
    tokenizer_name = "fhswf/BPE_GPT2_TinyStoriesV2_cleaned_2048"
    dataset_name = "fhswf/TinyStoriesV2_cleaned"

    tokenizer: Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log.info(tokenizer)

    training_data = get_training_data(dataset_name)
    d = next(training_data)
    log.info(len(d.split(" ")))
    log.info(d)
    e = tokenizer.encode(d)
    log.info(len(e))
    log.info(e)

def get_training_data(ds_name:str):
    dataset = load_dataset(ds_name)
    train_data = (dataset["train"][i]["text"] for i in range(0, len(dataset["train"])))
    test_data = (dataset["test"][i]["text"] for i in range(0, len(dataset["test"])))
    training_corpus = chain(train_data,test_data)
    return training_corpus

if __name__ == "__main__":
	run_standalone()