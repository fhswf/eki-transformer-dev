import qtransform as qt
from typing import List
from modelflow.command import common

class sbatch(common.SystemCommand):
    def __init__(self, *args):
        super().__init__(["sbatch"] + args)
        pass
    
class squeue(common.SystemCommand):
    def __init__(self, *args):
        super().__init__(["squeue"] + args)
        pass
    
class scancle(common.SystemCommand):
    def __init__(self, *args):
        super().__init__(["scancle"] + args)
        pass
    
# class srun(common.SystemCommand):
#     def __init__(self, script_path):
#         super().__init__(["sbatch", script_path])
#         pass
    
class sattach(common.SystemCommand):
    def __init__(self, *args):
        super().__init__(["sattach"] + args)
        pass
    
