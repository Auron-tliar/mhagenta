import asyncio
import logging
import time
import sys
from collections import Counter
from typing import Iterable, Any, Type, Coroutine
from pydantic import BaseModel
from pathlib import Path

import subprocess
from dill import dump
from threading import Lock
from mhagenta.utils import ModuleTypes, ModuleTypes, AgentCmd, ModuleConnector, AgentConnector, StatusReport, ModuleParams, LoggerExtras, ILogging
from mhagenta.params import *
from mhagenta.params.internal import *


def initialize_module(
        agent_params: AgentParams,
        module_params: ModuleParams,
        internal_params: BaseModel | None = None,
        path: Path | str = '.',
        merge_output: bool = False
) -> subprocess.Popen:
    if isinstance(path, str):
        path = Path(path)
    path = path.absolute()
    if path.is_dir():
        path = path / module_params.module_id
    path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        'agent_params': agent_params.model_dump(),
        'module_params': module_params.model_dump(),
        # 'module_params': module_params.__class__.dillable(**module_params.model_dump()),
    }
    if internal_params is not None:
        kwargs['internal_params'] = internal_params.model_dump()

    params = {
        'class': module_params.module_type,
        'kwargs': kwargs
    }
    with open(path, 'wb') as f:
        dump(params, f)

    return subprocess.Popen([
        f'{Path(sys.executable).absolute()}',
        f'{(Path(__file__).parent / "module_launcher.py").absolute()}',
        f'\"{path.absolute()}\"'],
        stdout=subprocess.PIPE if merge_output else None,
        stderr=subprocess.STDOUT if merge_output else None
    )


class HAgent(ILogging):
    AGENT_CONTROL_FREQ_RATIO: int = 2

    class ModuleData:
        def __init__(self, params: ModuleParams | None = None, internal_params: BaseModel | None = None):
            self.process: subprocess.Popen | None = None
            self.params: ModuleParams | None = params
            self.internal_params: BaseModel | None = internal_params
            self.ready: bool = False
            self.status: str = ''
            self.ts_status: float = -1.

    def __init__(self,
                 agent_id: str,
                 module_connector_cls: Type[ModuleConnector],
                 agent_connector_cls: Type[AgentConnector],
                 connector_kwargs: BaseModel | dict[str, Any] | None,
                 perceptors: Iterable[PerceptorParams] | PerceptorParams,
                 actuators: Iterable[ActuatorParams] | ActuatorParams,
                 ll_reasoner: LLParams,
                 learner: LearnerParams | None = None,
                 memory: MemoryParams | None = None,
                 knowledge: KnowledgeParams | None = None,
                 hl_reasoner: HLParams | None = None,
                 goal_graph: GGParams | None = None,
                 simulation_duration_sec: float = 60.,
                 step_frequency: float = 1.,
                 control_frequency: float | None = None,
                 status_period: int = 10,
                 start_time: float | None = None,
                 start_sync_delay: float = 2.,
                 save_dir: Path | str = './out/save',
                 log_level: int = logging.INFO,
                 log_format: str = f'[%(exec_time)f|%(mod_time)f|%(sim_time)s][%(levelname)s][%(agent_id)s][%(module_id)s] %(message)s'
                 ):
        self._init_time = time.time()
        self._agent_id = agent_id

        assert step_frequency > 0, 'step_frequency value should be positive!'
        assert status_period > 0, 'status_frequency should be a positive integer!'

        self._module_connector_cls = module_connector_cls

        self._start_time = start_time
        self._stop_time = -1.
        self._start_sync_delay = start_sync_delay
        self._sleep_coroutine: Coroutine | None = None
        self._stop_reason: str = ''
        self._lock = Lock()

        self._started = False

        if control_frequency is None:
            control_frequency = step_frequency / self.AGENT_CONTROL_FREQ_RATIO
        assert control_frequency > 0
        self._control_freq = control_frequency  / self.AGENT_CONTROL_FREQ_RATIO

        self._local_logger = logging.getLogger(f'{agent_id}.agent')
        self._local_logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        self._local_logger.addHandler(handler)

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir = save_dir.absolute()
        save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir = save_dir

        if isinstance(connector_kwargs, BaseModel):
            connector_kwargs = connector_kwargs.model_dump()
        self._connector_kwargs = connector_kwargs
        self._connector = agent_connector_cls(
            agent_id=self._agent_id,
            status_callback=self._on_module_status,
            log_func=self.log,
            **connector_kwargs
        )

        self._modules: dict[str, HAgent.ModuleData] = dict()

        perceptor_ids = self._add_modules(perceptors)
        actuator_ids = self._add_modules(actuators)

        ll_internal_params = LLInternalParams(
            perceptors=perceptor_ids,
            actuators=actuator_ids,
            has_learner=(learner is not None),
            has_memory=(memory is not None),
            has_hl_reasoner=(hl_reasoner is not None)
        )
        self._add_module(ll_reasoner, ll_internal_params)

        if learner is not None:
            self._add_module(learner)
        if memory is not None:
            memory_internal_params = MemoryInternalParams(
                has_hl_reasoner=(hl_reasoner is not None)
            )
            self._add_module(memory, memory_internal_params)
        if knowledge is not None:
            knowledge_internal_params = KnowledgeInternalParams(
                has_memory=(memory is not None)
            )
            self._add_module(knowledge, knowledge_internal_params)
        if hl_reasoner is not None:
            hl_internal_params = HLInternalParams(
                has_memory=(memory is not None)
            )
            self._add_module(hl_reasoner, hl_internal_params)
        if goal_graph is not None:
            self._add_module(goal_graph)

        self._simulation_duration = simulation_duration_sec

        self._agent_params = AgentParams(
            agent_id=agent_id,
            connector_cls=module_connector_cls,
            connector_kwargs=connector_kwargs,
            step_frequency=step_frequency,
            status_frequency=status_period,
            control_frequency=control_frequency,
            exec_start_time=self._init_time,
            stop_time=simulation_duration_sec,
            save_dir=str(save_dir),
            log_level=log_level,
            log_format=f'[%(exec_time)f|%(mod_time)f|%(sim_time)s][%(levelname)s][%(agent_id)s][%(module_id)s] %(message)s')

    def _add_module(self, params: ModuleParams, internal_params: BaseModel | None = None):
        if not params.module_id:
            params.module_id = params.module_type
        self._modules[params.module_id] = HAgent.ModuleData(params=params, internal_params=internal_params)

    def _add_modules(self, params: ModuleParams | Iterable[ModuleParams]) -> list[str]:
        if isinstance(params, ModuleParams):
            params = [params]
        ids = Counter()
        for param in params:
            if param.module_id in ids:
                param.module_id = f'{param.module_id}_{ids[param.module_id]}'
            ids[param.module_id] += 1
            self._add_module(params=param)

        return list(ids.keys())

    async def initialize_connection(self):
        await self._connector.initialize()

    async def start_connector(self):
        await self._connector.start()

    async def initialize_modules(self):
        for module_id, module_data in self._modules.items():
            module_data.process = initialize_module(
                agent_params=self._agent_params,
                module_params=module_data.params,
                internal_params=module_data.internal_params,
                path=self._save_dir
            )
        await self._wait_for_modules()

    async def start(self):
        if self._start_time is None:
            self.info(f'Start time not specified, starting now with {self._start_sync_delay}sec sync delay.')
            self._start_time = time.time() + self._start_sync_delay
        elif self._start_time < time.time():
            self.warning(f'Start time has passed, starting now with {self._start_sync_delay}sec sync delay.')
            self._start_time = time.time() + self._start_sync_delay
        elif self._start_time < time.time() + self._start_sync_delay:
            self.warning(f'Not enough time for full sync delay, will start in {self._start_sync_delay}sec to resolve.')
            self._start_time = time.time() + self._start_sync_delay
        else:
            self.info(f'Will start in {time.time() - self._start_time} seconds.')
        self._stop_time = self._start_time + self._simulation_duration
        self._started = True

        self.info(f'Starting the {self._simulation_duration}sec simulation with {self._start_sync_delay} offset delay...')
        self._cmd(cmd=AgentCmd.START, receiver_type=ModuleTypes.MODULE, start_time=self._start_time)

        await self._control_loop()

        self._stop_reason = 'timeout'
        await self._stop()

    async def _control_loop(self):
        while True:
            with self._lock:
                if time.time() < self._stop_time:
                    self._refresh_statuses()
                else:
                    break
            await asyncio.sleep(self._control_freq)

    def stop(self, reason: str = 'user input'):
        with self._lock:
            self._stop_reason = reason
            self._sleep_coroutine.close()

    async def _stop(self):
        self.info(f'Simulation stopping, reason: {self._stop_reason}...')

        self._cmd(AgentCmd.STOP, receiver_type=ModuleTypes.MODULE, reason=self._stop_reason)
        await self._wait_for_modules(False)

        for module in self._modules.values():
            module.process.wait()
        self.info(f'Agent finished execution!')

    def _cmd(self,
             cmd: str,
             receiver_type: ModuleTypes.LITERAL = ModuleTypes.ALL,
             module_names: str | Iterable[str] = ModuleTypes.ALL,
             module_ids: str | Iterable[str] = ModuleTypes.ALL, **kwargs):
        self.debug(f'Sending {cmd} command to {receiver_type}...')
        self._connector.cmd(
            cmd=AgentCmd(
                agent_id=self._agent_id,
                cmd=cmd,
                args=kwargs),
            receiver_type=receiver_type,
            module_names=module_names,
            module_ids=module_ids
        )

    @property
    def _logger_extras(self) -> LoggerExtras:
        ts = time.time()
        return LoggerExtras(
            exec_time=ts - self._init_time,
            mod_time=ts - self._init_time,
            sim_time=str(ts - self._start_time) if self._started else '-',
            agent_id=self._agent_id,
            module_id=ModuleTypes.AGENT.upper()
        )

    @property
    def _logger(self) -> logging.Logger:
        return self._local_logger

    def _on_module_status(self, status: StatusReport):
        self.debug(f'Received status: {status.model_dump()}')

        with self._lock:
            if status.status == status.READY:
                self._modules[status.module_id].ready = True
            elif status.status == status.FINISHED:
                self._modules[status.module_id].ready = False
            elif status.status == status.RUNNING:
                if self._modules[status.module_id].status == status.ERROR:
                    self.info(f'{status.module_id} has started responding!')

            self._modules[status.module_id].status = status.status
            self._modules[status.module_id].ts_status = time.time()

    def _refresh_statuses(self):
        for module_id, module in self._modules.items():
            if (module.status == StatusReport.RUNNING and
                    (time.time() - module.ts_status) / self._agent_params.control_frequency >= 2 * self._agent_params.status_frequency):
                module.status = StatusReport.ERROR
                self.warning(f'Module {module_id} is not responding...')

    async def _wait_for_modules(self, start: bool = True):
        while True:
            with self._lock:
                ready_flag = all([module.ready == start for module in self._modules.values()])
                if ready_flag:
                    break
            await asyncio.sleep(self._control_freq)
