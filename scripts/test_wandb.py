import wandb
import random
import os
from qtransform.utils import ID

WANDB_CACHE_DIR = os.environ.get("WANDB_CACHE_DIR", None)
if WANDB_CACHE_DIR is None:
    os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), "wandb_cache")
    
WANDB_CONFIG_DIR = os.environ.get("WANDB_CONFIG_DIR", None)
if WANDB_CONFIG_DIR is None:
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), "wandb_config")
    
WANDB_DIR = os.environ.get("WANDB_DIR", None)
if WANDB_DIR is None:
    os.environ["WANDB_DIR"] = os.path.join(os.getcwd())

# start a new wandb run to track this script
wandb.init(
    entity="eki-fhswf",
    # set the wandb project where this run will be logged
    project="qtransform test",
    name=ID,
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "None 2",
        "dataset": "NOne",
        "epochs": 10,
    }
)

import logging
import sys

logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stdout)  # Direct logs to stdout
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.info('This info will be captured by W&B.')
logger.error('This error will also be captured by W&B.')

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    print(f"{epoch=}")
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()