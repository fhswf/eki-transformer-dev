import qtransform as qt
from typing import List
from modelflow.command import common
from dataclasses import dataclass

@dataclass
class QTransformCommand(common.SystemCommand):
    cmd_bin: str = "python -m qtransform"

    #def __post_init__(self):
    #    self.cmd_bin = "python -m qtransform"
