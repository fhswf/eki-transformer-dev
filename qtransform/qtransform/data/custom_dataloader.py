from typing import Any
import torch
import numpy as np
from torch.utils import data
import glob


data.Sampler
class DynamicFileSystemDataset(data.Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.np_data = np.ndarray  ##TODO
        self._gather_files(file_path=file_path)
        
    def __len__(self) -> int:
        return len(self.samples)
        pass

    def __getitem__(self, index: int) -> Any:

        return img_tensor, class_id
    
    def _gather_files(self, file_path: str):
        self.file_path = file_path
        file_list = glob.glob(self.file_path + "*")
        for file in file_list:
            pass        
        pass
    