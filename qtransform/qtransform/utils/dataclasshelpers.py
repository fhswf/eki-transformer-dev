from typing import Any, Dict, Tuple, Union
import copy
import logging
from dataclasses import dataclass, fields, is_dataclass
log = logging.getLogger(__name__)

# only use dict for now; easy to add back later
def todict(obj) -> Dict:
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = todict(getattr(obj, f.name))
            result.update({f.name: value})
        return result
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[todict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        return type(obj)(todict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((todict(k), todict(v))
                         for k, v in obj.items())
    else:
        return copy.deepcopy(obj)

def fromdict(obj: Dict):
    # reconstruct the dataclass
    if is_dataclass(obj):
        result = {}
        for name, data in obj.items():
            result[name] = fromdict(data)
        return obj.type(**result)

    # exactly the same as before (without the tuple clause)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(fromdict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((fromdict(k), fromdict(v))
                         for k, v in obj.items())
    else:
        return copy.deepcopy(obj)
