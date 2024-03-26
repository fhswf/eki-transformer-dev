from qtransform.dataset import TokenizedDatasetGenerator, DatasetSplits, DatasetSplitType
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from typing import Union, Callable, Dict, List, Tuple
from omegaconf import DictConfig
from datasets import DatasetDict, Dataset, load_dataset
from datasets import config as hf_config
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import default_data_collator
from os.path import join
from os import makedirs
from itertools import chain
from logging import getLogger
import requests

log = getLogger(__name__)

#TODO: no check if tokenized but not grouped data exists
class HuggingfaceTokenizedDatasetGenerator(TokenizedDatasetGenerator):
    """
    TokenizedDatasetGenerator used to load huggingface datasets and tokenize datasets into arrow files.
    """
    #contains file extension and other distinguishing factors (e.g. block_size, tokenized or grouped..)
    #split is prepended to suffix (cache_file_prefix, split, DATASET_FILE_SUFFIX)
    DATASET_FILE_SUFFIX: str
    DUMP_FILE_PATH: str #path for intermediate result of tokenization (tokenized but not grouped)
    DATASET_FILE_PATH: str #by default: cache_dir from config
    API_URL: str #get split names from huggingface API endpoint

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.block_size = cfg.tokenized.args.block_size
        self.DATASET_FILE_SUFFIX = "grouped.arrow"
        self.API_URL = "https://datasets-server.huggingface.co/splits?dataset=" + self.cfg.name

        self.subset = self.cfg.name_args.get('subset', None)
        if self.subset is not None:
            self.DATASET_FILE_PATH = join(self.DATASET_FILE_PATH, self.subset, "")
            self.API_URL += "&config=" + self.subset
        log.debug(f'API_URL: {self.API_URL}\nDATASET_FILE_PATH: {self.DATASET_FILE_PATH}')
        self.DUMP_FILE_PATH = join(self.DATASET_FILE_PATH, "tokenized", "") #tokenized datasets are not grouped to block_size yet

        #make __post_init__?
        makedirs(self.DUMP_FILE_PATH, exist_ok=True)
        makedirs(self.DATASET_FILE_PATH, exist_ok=True)

    def get_tokenized_dataset(self) -> DatasetSplits:
        dataset_splits = DatasetSplits()
        for split, dataset_split_filename in self.CACHE_FILENAME_PREFIXES.items():
            try:
                dataset_splits[split] = Dataset.from_file(self.DATASET_FILE_PATH + dataset_split_filename + self.DATASET_FILE_SUFFIX)
            except FileNotFoundError as e:
                log.error(f'Split {split.name} not found locally.')
                raise e
        return dataset_splits

    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        #slice spits: https://huggingface.co/docs/datasets/loading#slice-splits
        #keys of DatasetDict are split types
        log.debug(f'Tokenizing data: {untokenized_data}')
        batch_size = self.cfg.untokenized.args.batches
        if batch_size is None:
            batch_size = 1000 #default for huggingface
        text_column_name = untokenized_data[list(untokenized_data.keys())[0]].column_names[0] #check if data is not in text column

        def tokenizer_function(batch):
            return {"input_ids": [tokenizer_singleton.tokenizer.encode(x) for x in batch[text_column_name]]}

        dump_file_names = {split.name: self.DUMP_FILE_PATH + self.CACHE_FILENAME_PREFIXES[split] + "tokenized.arrow" for split in DatasetSplitType}

        #tokenize them
        tokenized_datasets = untokenized_data.map(
            tokenizer_function,
            batched=True,
            #batch_size = batch_size,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset",
            cache_file_names=dump_file_names
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            log.critical(examples)
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

        cache_file_names = {split.name: self.DATASET_FILE_PATH + self.CACHE_FILENAME_PREFIXES[split] + self.DATASET_FILE_SUFFIX for split in DatasetSplitType}
        log.debug(f'tokenized_datasets: {tokenized_datasets}')

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            #batch_size=batch_size,
            desc=f"Grouping texts in chunks of {self.cfg.tokenized.args.block_size}",
            cache_file_names=cache_file_names
        )

    
    def get_split_mapping(self, split: DatasetSplitType) -> Tuple[str, float]:
        """
        Get the mapping of a split as specified in the untokenized dataset config.
        It includes the name of the original dataset split and the size if the split does not exist.
        """
        splits = self.cfg.untokenized.splits
        name = splits.names.get(split.name.lower())
        size = splits.sizes.get(split.name.lower())
        return (name, size)
    
    def get_hf_splits() -> List[str]:
        """
        Get names of splits from dataset without loading them into memory. 
        A hf API endpoint is used.
        """
        headers = {}
        response = requests.get(self.API_URL, headers=headers)
        if response.status_code != 200:
            log.error(f'Could not fetch splits from Huggingface API endpoint')
            return None
        data = response.json()
        return [split["split"] for split in data["splits"]] 

    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> DatasetDict:
        dataset_splits = {}

        for split in splits:
            split_name, split_size = self.get_split_mapping(split)
            try:
                dataset_split: Dataset = load_dataset(self.cfg.name, name=self.subset, split=split_name)
            except ValueError as split_not_found_error:
                log.warning(f'Subset "{self.subset}" for dataset "{self.cfg.name}" does not exist. Creating split from first split.')
                #TODO: dataset is loaded multiple times if splits do not exist
                dataset_split = load_dataset(self.cfg.name, name=self.subset)
                dataset_split = dataset_split[list(dataset_split.keys())[0]]
                dataset_split = dataset_split.train_test_split(split_size)[split_name]
            except FileNotFoundError as dataset_not_found_error:
                log.error(f'Dataset "{self.cfg.name}" with subset: "{self.subset}" does not exist.')
                raise FileNotFoundError()
            #TODO: config flag column_name not used, is it necessary?
            text_column_name = "text" if "text" in dataset_split.column_names else dataset_split.column_names[0]
            dataset_split = dataset_split.select_columns(text_column_name)
            dataset_splits[split.name] = dataset_split #from eval to EVAL for easy enum support
        return DatasetDict(dataset_splits)

    def get_collator(self) -> Callable:
        return DataCollatorWithPadding(tokenizer=tokenizer_singleton.tokenizer, padding='max_length', max_length=self.cfg.tokenized.args.block_size)