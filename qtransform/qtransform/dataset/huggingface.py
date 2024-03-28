from qtransform.dataset import TokenizedDatasetGenerator, DatasetSplits, DatasetSplitType
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform.tokenizer import TransformersTokenizer
from typing import Union, Callable, Dict, List, Tuple, Optional, Mapping
from omegaconf import DictConfig
from datasets import DatasetDict, Dataset, load_dataset
from datasets import config as hf_config
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers.utils.generic import TensorType
from transformers.tokenization_utils_base import EncodedInput, PaddingStrategy, BatchEncoding
from os.path import join
from os import makedirs
from itertools import chain
from logging import getLogger
import requests
from dataclasses import dataclass

log = getLogger(__name__)

@dataclass
class SplitConfig():
    split: str
    mapping: str
    size: float
    exists: bool

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
        text_column_name = untokenized_data[list(untokenized_data.keys())[0]].column_names[0] #check in what column name the data is stored

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
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            block_size = self.cfg.tokenized.args.block_size
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

        cache_file_names = {split.name: self.get_filepath_split(split) for split in DatasetSplitType}
        log.debug(f'tokenized_datasets: {tokenized_datasets}')

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            #batch_size=batch_size,
            desc=f"Grouping texts in chunks of {self.cfg.tokenized.args.block_size}",
            cache_file_names=cache_file_names
        )
        log.debug(f'Grouped datasets: {lm_datasets}')
        #log.debug(f'First sample: {len(lm_datasets[DatasetSplitType.EVAL.name]["input_ids"][0])}') #make sure it is the length of block_size

    
    def get_split_mapping(self, split: DatasetSplitType) -> SplitConfig:
        """
        Get the mapping of a split as specified in the untokenized dataset config.
        It includes the name of the original dataset split and the size if the split does not exist.
        """
        if not isinstance(split, DatasetSplitType):
            log.error(f'Invalid split: {split}')
            raise TypeError()
        return SplitConfig(**self.cfg.untokenized.splits.get(split.name.lower()))
    
    def get_hf_splits(self) -> Tuple[int, Union[List[str], None]]:
        """
        Get names of splits from dataset without loading them into memory. 
        A hf API endpoint is used (https://huggingface.co/docs/datasets-server/splits).
        
        Arguments:
            None 
        
        Returns:
            A Tuple of:
            - HTTPS status code of the request to huggingface API
            - A list of split names if status code was 200, otherwise return None
        """
        headers = {}
        response = requests.get(self.API_URL, headers=headers)
        if response.status_code != 200:
            splits = None
        else:
            data = response.json()
            splits = [split["split"] for split in data["splits"]]
        return (response.status_code, splits)

    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> DatasetDict:
        dataset_splits = {}
        log.debug(f'Getting dataset: {self.cfg.name} {self.subset if self.subset is not None else ""}')
        status, fetched_splits = self.get_hf_splits()
        #check if dataset, subset and splits exist at all
        if fetched_splits is None:
            log.error(f'Could not fetch splits from Huggingface API endpoint, either because the specified config has errors or because the repository is private.' \
                f' (Status code: {status})')
            raise RuntimeError()
        for split in splits:
            split_cfg: SplitConfig = self.get_split_mapping(split)
            log.debug(f'Getting split: {split_cfg.mapping}')
            try:
                dataset_split = load_dataset(self.cfg.name, name=self.subset, split=split_cfg.mapping)
                if not split_cfg.exists:
                    #alternative: load_dataset("name[:10%]") gets first 10 percent of dataset. not random though
                    log.info(f'Creating split: "{split_cfg.split}" from "{split_cfg.mapping}" with size: {split_cfg.size}.')
                    #we only have the split name of the mapped split (e.g. train), meaning that we take that split and reduce its size
                    dataset_split = dataset_split.train_test_split(1-split_cfg.size)[split_cfg.mapping]
            except ValueError as split_not_found_error:
                log.error(f'Split "{split.split}" does not exist with "exists: {split_cfg.exists}", "mapping: {split_cfg.mapping}". Maybe check mapping?')
                raise split_not_found_error
            except FileNotFoundError as dataset_not_found_error:
                log.error(f'Dataset "{self.cfg.name}" with subset: "{self.subset}" does not exist.')
                raise dataset_not_found_error
            #TODO: config flag column_name not used, is it necessary?
            text_column_name = "text" if "text" in dataset_split.column_names else dataset_split.column_names[0]
            dataset_split = dataset_split.select_columns(text_column_name)
            dataset_splits[split.name] = dataset_split #from eval to EVAL for easy enum support
        return DatasetDict(dataset_splits)

    def get_collator(self) -> Callable:
        tokenizer = tokenizer_singleton.tokenizer
        #DataCollatorWithPadding pads with tokenizer's pad method which is primarily implemented by hf's tokenizers
        if not isinstance(tokenizer, TransformersTokenizer):
            log.debug(f'Setting custom padding function')
            #setattr(tokenizer, 'pad', pad)
        else:
            #no need as transformers tokenizers have pad method
            tokenizer = tokenizer_singleton.tokenizer.tokenizer
        #TODO: returns dict of structure: {input_ids: tokens, labels: tokens}
        return DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=self.cfg.tokenized.args.block_size)
        



def pad(self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        #ignored but kept here to avoid param errors
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
    """
    This function is primarily used and implemented to support Huggingface DataCollators 
    when specifying a custom collate_fn within the torch Dataloader. As such, large chunks of 
    PreTrainedTokenizerBase's pad methods are copied.
    For our use case, the inputs are padded to a maximum length (context length of the model). The maximum length is specified
    by max_length.
    The column name of the encoded inputs is set to "input_ids" and "labels" for now.
    The following is an excerpt from huggingface's pad comment:
    
    Pad a single encoded input or a batch of encoded inputs up to predefined length.
    
    """
    REQUIRED_INPUT = "input_ids" #name of column where tokens are saved
    assert return_tensors == 'pt' #pytorch tensors
    if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
    required_input = encoded_inputs[REQUIRED_INPUT]
    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
        # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
        for item in required_input:
            if len(item) != 0:
                first_element = item[0]
                break
    #actually do padding, from _pad 
    difference = max_length - len(required_input)
    log.debug(f'DIFFERENCE: {difference}\nENCODED_INPUTS: {encoded_inputs}\nREQUIRED_INPUT: {required_input}')
    encoded_inputs[REQUIRED_INPUT] = required_input + [self.PADDING_TOKEN] * difference
    return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
