import asyncio
import os
from pathlib import Path
from sys import argv, exit
from typing import *

import dill
from pydantic import BaseModel

from mhagenta.modules import *
from mhagenta.states import *
from mhagenta.utils import ModuleTypes, Observation, ActionStatus
from mhagenta.core.processes import run_agent_module, MHAModule, ModuleBase, GlobalParams
from mhagenta.bases import *


class ModuleSet(BaseModel):
    module_cls: Type[MHAModule]
    base: Type[ModuleBase]


MODULE_NAME_TO_CLASS = {
    ModuleTypes.ACTUATOR: ModuleSet(module_cls=Actuator, base=ActuatorBase),
    ModuleTypes.PERCEPTOR: ModuleSet(module_cls=Perceptor, base=PerceptorBase),
    ModuleTypes.LLREASONER: ModuleSet(module_cls=LLReasoner, base=LLReasonerBase),
    ModuleTypes.LEARNER: ModuleSet(module_cls=Learner, base=LearnerBase),
    ModuleTypes.MEMORY: ModuleSet(module_cls=Memory, base=MemoryBase),
    ModuleTypes.KNOWLEDGE: ModuleSet(module_cls=Knowledge, base=KnowledgeBase),
    ModuleTypes.HLREASONER: ModuleSet(module_cls=HLReasoner, base=HLReasonerBase),
    ModuleTypes.GOALGRAPH: ModuleSet(module_cls=GoalGraph, base=GoalGraphBase)
}


if __name__ == "__main__":
    if len(argv) < 2:
        exit('Expected [params_path] as an argument!')

    file_path = Path(argv[1].replace('\"', ''))
    with open(file_path, 'rb') as f:
        params = dill.load(f)
    os.remove(file_path)
    module_name, params = params['class'], params['kwargs']
    module_data = MODULE_NAME_TO_CLASS[module_name]
    module_cls = module_data.module_cls
    params['base'] = dill.loads(params['base'])
    params['global_params'] = GlobalParams(**params['global_params'])

    exit_reason = asyncio.run(run_agent_module(module_cls, **params))

    print(f'Module {params["base"].module_id} exited, reason: {exit_reason}')
