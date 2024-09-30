import os
from typing import Any
from qtransform.utils.helper import get_output_dir, singleton
from .adjectives import values as adj
from .nouns import values as nouns
import random
import datetime
import logging
import glob

log = logging.getLogger(__name__)
random.seed(datetime.datetime.now())

@singleton()
class ID(object):
    # singelton done by decorator, otherwise use this:
    # _instance = None
    # def __new__(cls, *args, **kwargs):
    #     # single object instance as class attr
    #     if cls._instance is None:
    #         cls._instance = super(ID, cls)(*args, **kwargs) # get instance and call init on class
    #     return cls._instance
    _max_gen_tries:int = 10
    
    def __init__(self) -> None:
        self.timestamp = datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
        self.name = generate_free_name()
        self.id = str(self.timestamp) + "-" + str(self.name)
        pass
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_id(self):
        return self.id
    
    def __call__(self):
        return self.get_id()
    
    def get_name(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.get_id()
    
    def __str__(self) -> str:
        return self.get_id()

def generate_free_name() -> str:
    """checks f"{cwd}/{output dir()}" for existing name combinations and return when it finds a random free name.

    Returns:
        str: generated name by generate_name()
    """
    tries:int = 1 # start by one as this is a do while loop

    name = generate_name()
    while not name_is_free(id):
        tries = tries + 1
        name = generate_name()
        # just use a name, ids are unique anyway due to timestamps
        if tries >= ID._max_gen_tries:
            break

    return name

def generate_name():
    """creates a random name.

    Returns:
        str: "adjective-noun"
    """
    return f"{random.choice(adj)}-{random.choice(nouns)}"

def name_is_free(id:str) -> bool:
    """checks wether id has been used in output dir.
    
    Args:
        id (str): "adjective-noun"

    Returns:
        bool: true if id has not been found in output dir
    """
    return len(glob.glob(f"{get_output_dir()}{os.path.sep}**{id}*")) == 0