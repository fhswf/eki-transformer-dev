from dataclasses import dataclass
from modelflow.command.common import Command

@dataclass
class LayerExchange(Command):
    source_layer: str
    targer_layer: str

    