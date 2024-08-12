import dill
from sys import argv, exit
import asyncio
from typing import *
from pydantic import BaseModel
from mhagenta.modules import *
from mhagenta.utils import ModuleTypes
from mhagenta.core.processes import run_agent_module, MHAModule, ModuleBase, GlobalParams
from mhagenta.base import *


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

    with open(argv[1].replace('\"', ''), 'rb') as f:
        params = dill.load(f)
    module_name, params = params['class'], params['kwargs']
    module_data = MODULE_NAME_TO_CLASS[module_name]
    module_cls = module_data.module_cls
    params['base'] = dill.loads(params['base'])  # module_data.base(**dill.loads(params['base']))
    params['global_params'] = GlobalParams(**params['global_params'])

    exit_reason = asyncio.run(run_agent_module(module_cls, **params))

    print(f'Module {params["base"].module_id} exited, reason: {exit_reason}')