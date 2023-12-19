from logging import getLogger
from torch import device, cuda, backends
log = getLogger(__name__)

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