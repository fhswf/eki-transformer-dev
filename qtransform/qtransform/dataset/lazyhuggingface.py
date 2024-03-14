from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)

import os
from itertools import chain
from qtransform.dataset import DatasetSplits, DatasetWrapper

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from datasets import config as hf_config
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import default_data_collator


#https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes
class LazyHuggingfaceDatasetWrapper(DatasetWrapper):
    """
        Retrieves a huggingface datasetand returns a DatasetInfo object. Under the hood, the datasets are tokenized and written
        into a numpy memmap file on the local user's harddrive for performance reasons. It also avoids having to load and tokenize 
        the same datasets multiple times.
        When tokenizer is givin during construction the givin tokenizer will be usesd, otherwise the toknizer will be infered from the hydra config.
    """
    def __init__(self, cfg: DictConfig, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(cfg)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print(f"{cfg}")
        self.tokinzer_name = cfg.tokenizer.get("name")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name, use_fast=False)
            self.tokenizer = tokenizer
        else: 
            self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        #https://github.com/huggingface/datasets/issues/3638
        self.tokenizer("Call init Tokenier", "to enable cacheing bug", truncation=True)

        self.max_block_size = self.tokenizer.model_max_length

    def get_loader(self, split: str) -> DataLoader:
        log.debug(f"get_loader config: {self.cfg.dataloader} for split {split}")
        # loader = DataLoader(data, generator=torch.Generator(device='cuda'), **dataloader_cfg) # does no work for dataloader forks
        kwargs = {**self.cfg.dataloader}
        if split=='train':
            kwargs['shuffle'] = False
        if self.cfg.get('collate_fn'):
            log.warning("TODO collate_fn via config is not supported yet")

        kwargs['collate_fn'] = self.data_collator
        if split not in self.datasets.__dict__.keys():
            raise KeyError(f"Split {split} not found in avaiable dataset splits. Usually train eval or bench.")
        loader = DataLoader(dataset=self.datasets[split], **kwargs)
        log.debug(f'len of dataset loader: {len(loader)}')
        return loader
    
    def load_dataset(self, block_size=None, batch_size=None):
        if block_size is None:
            block_size = self.max_block_size
        log.info(f"Block size = seq length {block_size}")
        
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding='max_length', max_length=block_size)

        if batch_size is None:
            batch_size = self.cfg.dataloader.batch_size

        # workaround for tokinzer cache invalidation because hash of tokinzer changes for no reason
        hf_cache_file_name =  "cache-"  + self.tokinzer_name.replace('/', '__')  + "-" + str(block_size) + "-"
        hf_cache_file_name = hf_cache_file_name.lower()
        if self.cfg.get('subset') is not None:
            dataset_name = self.cfg.name + "_" +  self.cfg.get('subset')
        else:
            dataset_name = self.cfg.name
        dataset_name = dataset_name.replace('/', '__').lower()
        cache_file_path = os.path.join(hf_config.HF_DATASETS_CACHE, "qtransform_tokenized" , dataset_name)
        cache_file_full_prefix = os.path.join(cache_file_path,  hf_cache_file_name)
        log.info(f"dataset cache_file_prefix for tokenizer set to {cache_file_full_prefix}")
        os.makedirs(cache_file_path, exist_ok=True)

        # load dataset to disk (might trigger expensive downloads)
        log.info(f"Beginn loading HF dataset. This might take a while")
        dataset = load_dataset(self.cfg.name, name=self.cfg.get('subset'))
        # configure padding and batching via data_collate and grouping function
        # Note that this might be done differently in other DatasetWrappers.
        # Also this might be more ressource intensive because of huddingface dataloader workers being a **** sometimes  
        log.info(f"Setting padding and batching via data_collate and grouping function")
        dataset = self.map_dataset(dataset, block_size, batch_size, cache_file_prefix=cache_file_full_prefix)
        log.debug(f" Dataset containes {dataset}")

        if isinstance(dataset, Dataset):
            log.warning(f"Dataset {self.cfg.name}, subset {self.cfg.get('subset')} has no splits. Setting all splits the same")
            self.datasets.train = dataset
            self.datasets.eval = dataset
            self.datasets.bench = dataset
        elif isinstance(dataset, dict):
            if self.cfg.get('splits') is not None:
                for k,v in self.cfg.splits.items():
                    log.debug(f"Mapping dataset our key {k} to found key {v} of external dataset")
                    self.datasets[k] = dataset[v]
            else:
                log.warning("Please use splits mapping. Trying to infer splits with common names")
                try:
                    self.datasets.train = dataset['train']
                    self.datasets.eval = dataset['validation']
                    self.datasets.bench = dataset['test']
                except Exception as e:
                    log.error(f"Infer of common dataset splits failed, Error was {e}", exc_info=1)
                    raise e
        else:
            log.error(f"Huggingface dataset:  load_dataset return  for {self.cfg.name}, subset {self.cfg.get('subset')} has unsupported type. Type was {type(dataset)}. Needs to be dict with names")
            raise KeyError

        pass
    

    def map_dataset(self, dataset:DatasetDict, block_size, batch_size, cache_file_prefix):
        """apllies mapping and tokenizer do dataset"""

        log.info(f"Dataset has  column names: {dataset.column_names}. Only using \'text\' and cutting all other")
        if isinstance(dataset, dict) and 'train' in dataset.keys():
            text_column_name = "text" if "text" in dataset['train'].column_names else dataset['train'].column_names[0]
            column_names = dataset['train'].column_names
        else:
            text_column_name = "text" if "text" in dataset.column_names else dataset.column_names[0]
            column_names = dataset.column_names
            
        def tokenize_function(examples):
            return self.tokenizer(examples[text_column_name])

        cache_file_names = {
          'test':  cache_file_prefix + "tokenized-" + 'test.arrow',
          'train': cache_file_prefix + "tokenized-" + 'train.arrow',
          'validation': cache_file_prefix +  "tokenized-" + 'validation.arrow'
        }
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            cache_file_names=cache_file_names
        )
        
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        cache_file_names = {
          'test':  cache_file_prefix + "grouped-" + 'test.arrow',
          'train': cache_file_prefix + "grouped-" + 'train.arrow',
          'validation': cache_file_prefix +  "grouped-" + 'validation.arrow'
        }

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
            cache_file_names=cache_file_names
        )

        return lm_datasets
