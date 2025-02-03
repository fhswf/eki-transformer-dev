import abc
from typing import Any, Dict, ItemsView, KeysView
from modelflow.utils.helper import singleton
import logging
log = logging.getLogger(__name__)

class Store(abc.ABC):
    # TODO make this configigurable with hydra config
    """
    Base class for a variable Store. Holds inputs and outputs of commands and pipeline steps.
    Observers can be registered to listen to key updates. Not a dataclass so we can put multiple decorators around subclasses.
    """
    def __init__(self):
        # Holds any data, accessable by str keys, should always stay this way 
        self._data: Dict[str, Any] = {}
        self._observer: Dict[str, Any] = {}
        
    def update(self, key, value):
        self._data[key] = value
        self._notify(key, value)

    def keys(self) -> KeysView[Any]:
        return self._data.keys()
    
    def items(self) -> ItemsView[Any, Any]:
        return self._data.items()
    
    def __contains__(self, value) -> bool:
        return self.__dict__.keys().__contains__(value)
    
    def get(self, key, default) -> Any:
        return self._data.get(key, default=default)
    
    def _notify(self, key, value):
        if key in self._observers:
            for observer in self._observers[key]:
                observer.update(key, value)
    
    def register_observer(self, key, observer):
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(observer)

    def unregister_observer(self, key, observer):
        if key in self._observers:
            self._observers[key].remove(observer)
            if not self._observers[key]:
                del self._observers[key]


@singleton()
class OutputManager(Store):
    """
    Holds Information about Command Outputs and notifies outputs to subscribed keys in dict when a new value gets written to the output.
    """
    def __init__(self):
        super().__init__()
        log.info("creating output store")
        pass