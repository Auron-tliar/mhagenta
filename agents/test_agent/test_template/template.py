from pydantic import BaseModel
from typing import Any, Iterable, Type
from mhagenta.utils import ModuleConnector, AgentConnector
from mhagenta.params import *


class TestTemplate(BaseModel):
    agent_id: str
    module_connector_cls: Type[ModuleConnector]
    agent_connector_cls: Type[AgentConnector]
    connector_kwargs: BaseModel | dict[str, Any] | None = None
    perceptors:PerceptorParams
    actuators:ActuatorParams
    ll_reasoner: LLParams
    learner: LearnerParams | None = None
    memory: MemoryParams | None = None
    knowledge: KnowledgeParams | None = None
    hl_reasoner: HLParams | None = None
    goal_graph: GGParams | None = None
    simulation_duration_sec: float = 20.
    step_frequency: float = 1.
    control_frequency: float = .5
    status_period: int = 10
    start_time: float | None = None
    start_sync_delay: float = 2.
    save_dir: str = 'D:\\bsc-tmp\\phd\\test\\out\\save'
    verbose: bool = True
