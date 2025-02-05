from modelflow.command.common import Task, TaskInterator, Command
from modelflow.scheduler.common import Serializable
from dataclasses import dataclass
from typing import Any
import logging
import torch
from modelflow.store.store import OutputManager

log = logging.getLogger(__name__)


@dataclass    
class LayerExchange(TaskInterator, Serializable):
    # TODO only works for qtransform atm 
    exchange_loop_cmd : str = ""
    source_layer: str = ""
    target_layer: str = ""
    strategy: str = "first" # TODO replace this with enum or class
    until: Any  = None # TODO replace this with enum or class
    current_step: Any = None
    init_cmd: str = "" 
    def __post_init__(self):
        # self.tasks = [] # done in super
        super().__post_init__()
    
    def maybe_create_sub_task(self)->Task:
        """create a sub tasks for a conversion step and fill missing parts of sub tasks with task content of this taskiterators common attr"""
        if self.check_conversion_done():
            self.create_sub_task()
    
    def create_sub_task(self):
        """create a sub task for a conversion step"""
        if self.current_step == 0 and len(self.init_cmd) > 0:
            cmd = self.init_cmd
        else:
            cmd = self.exchange_loop_cmd
        
        sub_task = Command(
            cmd=cmd,
            outputs=self.outputs,
            name=f"ConversionStep{self.current_step}",
        )
        self.tasks.append(sub_task)
        self.current_step = self.current_step + 1
        log.info(f"Created sub task: {sub_task}")
            
    def get_checkpoint_location(self):
        """query the checkpoint location from last command output from OutputManager. key is run_chkpt"""
        output_manager = OutputManager()
        checkpoint_location = output_manager.get("run_chkpt")
        
        if checkpoint_location:
            log.info(f"Checkpoint location: {checkpoint_location}")
            return checkpoint_location
        else:
            log.warning("Checkpoint location not found in output manager.")
            return None # we dont have a checlpoint location at the first run
            # raise KeyError("Checkpoint location not found in output manager.")
        
    def parse_torch_model(self):
        # parse model to get layer information
        if self.get_checkpoint_location() is None:
            return 1 # assume we have at least one layernorm layer
        checkpoint = torch.load()
        model = checkpoint['model']

        # Dynamically get the layer class from the string
        layer_class = getattr(torch.nn, self.source_layer, None)
        if layer_class is None:
            log.error(f"Layer class {self.source_layer} not found in torch.nn")
            return 0

        # Count the number of specified layers
        layer_count = sum(1 for layer in model.modules() if isinstance(layer, layer_class))
        
        log.info(f"Number of {self.source_layer} layers: {layer_count}")
        return layer_count
    
    def replace_layer(self):
        # replace source_layer in model with target_layer
        if self.get_checkpoint_location() is None:
            log.warning("No checkpoint location found. Cannot replace layers.")
            return
        
        checkpoint = torch.load(self.get_checkpoint_location())
        model = checkpoint['model']

        # Dynamically get the layer classes from the strings
        source_layer_class = getattr(torch.nn, self.source_layer, None)
        target_layer_class = getattr(torch.nn, self.target_layer, None)
        
        if source_layer_class is None:
            log.error(f"Source layer class {self.source_layer} not found in torch.nn")
            return
        
        if target_layer_class is None:
            log.error(f"Target layer class {self.target_layer} not found in torch.nn")
            return

        # Find all instances of the source layer
        layers = []
        def find_layers(module):
            for name, child in module.named_children():
                if isinstance(child, source_layer_class):
                    layers.append((module, name))
                else:
                    find_layers(child)
        
        find_layers(model)

        if not layers:
            log.warning(f"No layers of type {self.source_layer} found in the model.")
            return

        # Replace the first or last layer based on configuration
        if self.strategy == "first":
            module, name = layers[0]
        elif self.strategy == "last":
            module, name = layers[-1]
        else:
            log.error(f"Unknown strategy {self.strategy}. Use 'first' or 'last'.")
            return

        # TODO figure out ou tou tto do tis smartly, where layer information get preserved when possible
        setattr(module, name, target_layer_class())
        log.info(f"Replaced {self.source_layer} with {self.target_layer} in {name}")

        # Save the modified model back to the checkpoint
        checkpoint['model'] = model
        torch.save(checkpoint, self.get_checkpoint_location())
        log.info("Model layers replaced and checkpoint saved.")
        
    def check_conversion_done(self):
        if self.get_checkpoint_location() is None and (self.init_cmd is None or len(self.init_cmd) == 0):    
            raise Exception("No checkpoint location found and no init_cmd provided to create one.")
        else:
            log.info("No init_cmd found, but checkpoint exists. ignoring init_cmd.")
            
        layers_to_do = self.parse_torch_model() > 0
        return not layers_to_do
        
    def get_save_attributes(self):
        return ["source_layer", "target_layer", "strategy", "until", "current_step"]

    def __before_task__(self):
        # check if conversion was completed by checking the saved model, or the output logs
        self.maybe_create_sub_task()