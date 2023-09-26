from torch.utils.data import DataLoader
from omegaconf import DictConfig

def get_data(cfg: DictConfig) -> DataLoader:
    """ takes a config and returns a ready to use torch compatible dataloader/dataset."""
    # dataloader = DataLoader(num_workers=cfg.run.data.num_workers, batch_size=cfg.run.data.batch_size, shuffle=cfg.run.data.shuffle_data, **cfg.data)
    
    match cfg.module:
        case "":
            pass
        case _:
            pass
    # ==> use dict here
    return