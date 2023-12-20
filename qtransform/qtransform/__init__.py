from logging import getLogger
import os
from torch import device, cuda, backends
log = getLogger(__name__)

def get_module_config_path():
    return os.path.join('/'.join(__file__.split('/')[:-2]), 'qtransform' , 'conf')

def main(cfg):
    """Run this app like amodule, Note: cfg is a Hydra config (OmegaConf Object)"""
    from qtransform import  __main__ as mn
    mn.main(cfg)

def notebook_run(args):
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import qtransform 
    with initialize_config_dir(version_base=None, config_dir=qtransform.get_module_config_path()):
        cfg = compose(config_name="config.yaml", overrides=args)
        print(cfg)
        main(cfg)


class DeviceSingleton:
    """
        Boilerplate class which contains the device used for the entire process. The reason why a class is created is in order to monitor when changes
        to the device are going to occur by using the property decorator. Currently, changes will be allowed and logged.
    """
    def __init__(self):
        self._device = None
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_device):
        match new_device:
            case 'cuda':
                new_device = 'cuda' if cuda.is_available() else 'cpu'
            case 'gpu':
                new_device = 'cuda' if cuda.is_available() else 'cpu'
            case 'mps':
                new_device = 'mps' if backends.mps.is_available() else 'cpu'
            case 'cpu':
                new_device = 'cpu'
            case _:
                log.warning(f'Device {new_device} not recognized. Using default: CPU')
                new_device = 'cpu'
        self._device = device(new_device)
        log.info(f'Using device: {new_device}')


device_singleton = DeviceSingleton()