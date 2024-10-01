from itertools import chain
import logging

from tokenizers import Tokenizer
log = logging.getLogger(__name__)

from transformers import AutoTokenizer
from datasets import load_dataset

import qtransform
args = [ "run=script", "run.file=none" ]
@qtransform.with_config(args, logging.DEBUG)
def run_standalone(cfg):
    log.info(f"running {__file__}")
    
    tokenizer_base = "EleutherAI/gpt-neo-125M"
    dataset_name = "fhswf/TinyStoriesV2_cleaned"

    old_tokenizer: Tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)

    training_data = get_training_data(dataset_name)
    old_tokenizer.eos_token = "<|endoftext|>"

    tokenizer = old_tokenizer.train_new_from_iterator(training_data, 1024)
    tokenizer.save_pretrained("code-fhswf/BPETinyStoriesV2_cleaned_v3")


def get_training_data(ds_name:str):
    dataset = load_dataset(ds_name)
    train_data = (dataset["train"][i]["text"].rstrip("<|endoftext|>") for i in range(0, len(dataset["train"])))
    test_data = (dataset["test"][i]["text"].rstrip("<|endoftext|>") for i in range(0, len(dataset["test"])))
    training_corpus = chain(train_data,test_data)
    return training_corpus

if __name__ == "__main__":
	run_standalone()