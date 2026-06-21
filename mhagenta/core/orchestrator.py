import asyncio
import logging
import os
import shutil
import sys
import time
import warnings
import inspect
import sysconfig
from asyncio import TaskGroup
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, ClassVar, cast
from collections.abc import Iterable, Callable
import functools
import dateutil.parser
import dateutil.tz

import dill
import docker
import pika
from pika.adapters import BlockingConnection
from pika.exceptions import AMQPConnectionError
from docker.errors import NotFound
from docker.models.containers import Container
from docker.models.images import Image

import mhagenta
from mhagenta.bases import *
from mhagenta.containers import *
from mhagenta.core.connection import Connector, RabbitMQConnector
from mhagenta.utils import DEFAULT_PORT, DEFAULT_RMQ_IMAGE
from mhagenta.utils.common import DEFAULT_LOG_FORMAT, Directory
from mhagenta.environment import MHAEnvBase
from mhagenta.gui import Monitor
from mhagenta.utils.common.classes import EDirectory


@dataclass
class Entry:
    kwargs: dict[str, Any]
    dir: Path | None = None
    tags: Iterable[str] | None = None
    image: Image | None = None
    containers: dict[str, Container] | None = None
    port_mapping: dict[int, int] | None = None
    script_path: Path | None = None
    requirements_path: Path | None = None
    extra_runtime_sources: tuple[Path, ...] = tuple()


@dataclass
class AgentEntry(Entry):
    agent_id: str = field(kw_only=True)
    save_dir: Path | None = None
    num_copies: int = 1

    _flat_modules: list[ModuleBase] = field(default_factory=list)

    _MODULE_KEYS: ClassVar[tuple[str, ...]] = ('perceptors', 'actuators', 'll_reasoners', 'learners',
                                               'knowledge', 'hl_reasoners', 'goal_graphs', 'memory')

    @property
    def modules(self) -> list[ModuleBase]:
        if self._flat_modules:
            return self._flat_modules

        for key in self._MODULE_KEYS:
            value = self.kwargs.get(key, None)
            if value is None:
                continue
            if isinstance(value, ModuleBase):
                self._flat_modules.append(value)
            elif isinstance(value, Iterable):
                for module in value:
                    if not isinstance(module, ModuleBase):
                        raise TypeError(f'Expected {key} to be a ModuleBase instance or a collection of them, got {type(module)}')
                    self._flat_modules.append(module)
            else:
                raise TypeError(f'Expected {key} to be a ModuleBase instance or a collection of them, got {type(value)}')
        return self._flat_modules

    @property
    def module_ids(self) -> list[str]:
        return [module.module_id for module in self.modules]

    def __hash__(self) -> int:
        return hash(self.agent_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AgentEntry):
            return False
        return self.agent_id == other.agent_id


@dataclass
class EnvironmentEntry(Entry):
    env_id: str = field(kw_only=True)
    address: dict[str, Any] = field(kw_only=True)

    @property
    def base(self) -> MHAEnvBase:
        return self.kwargs['base']

    def __hash__(self) -> int:
        return hash(self.env_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EnvironmentEntry):
            return False
        return self.env_id == other.env_id


class LogParser:
    """
    Utility class for parsing logs from several containers and outputting them in order of their timestamps.
    """
    US = timedelta(microseconds=1)

    @dataclass
    class SourceInfo:
        last_ts: datetime
        path: Path | None = None
        container: Container | None = None

    @functools.total_ordering
    @dataclass
    class LogEntry:
        ts: datetime
        msg: str
        source: 'LogParser.SourceInfo'

        def __gt__(self, other: 'LogParser.LogEntry') -> bool:
            return self.ts > other.ts

        def __eq__(self, other: 'LogParser.LogEntry') -> bool:
            return self.ts == other.ts

    def __init__(
            self,
            stop_checker: Callable[[], bool],
            check_freq: float = 1.,
            save_logs: os.PathLike | str | None = None,
            no_stdout: bool = False
    ) -> None:
        self._sources: dict[str, LogParser.SourceInfo] = dict()
        self._check_freq: float = check_freq
        self._stop_checker: Callable[[], bool] = stop_checker
        self._save_logs: Path | None = Path(save_logs) if save_logs else None
        self._stdout: bool = not no_stdout

        self._init_ts = datetime.now()

    def add_container(self, source: AgentEntry | EnvironmentEntry) -> None:
        sids: list[str]
        if isinstance(source, EnvironmentEntry):
            sids = [source.env_id]
        elif isinstance(source, AgentEntry):
            assert source.containers
            if len(source.containers) == 1:
                sids = [source.agent_id]
            else:
                sids = list(source.containers.keys())
        else:
            raise ValueError('Cannot use general Entry with the LogParser!')

        for sid in sids:
            self._sources[sid] = self.SourceInfo(
                last_ts=self._init_ts,
                path=(self._save_logs / f'{sid}.log') if self._save_logs is not None else None,
                container=source.containers[sid]
            )

    @staticmethod
    def _add_log(log: str | bytes, save_path: Path | None = None, print_to_stdout: bool = True) -> None:
        if isinstance(log, bytes):
            log = log.decode().strip('\n\r')
        if '[status_upd]' in log:
            return
        if print_to_stdout:
            print(log)
        if save_path is not None:
            with open(save_path, 'a') as f:
                f.write(f'{log}\n')

    async def run(self) -> None:
        logs: list[LogParser.LogEntry] = list()
        log_lines: list[str]

        while True:
            if self._stop_checker():
                break
            for sid, info in self._sources.items():
                assert info.container is not None
                raw_log = info.container.logs(stdout=True, stderr=True, tail='all', timestamps=True, since=(info.last_ts + self.US).timestamp())
                log_lines = raw_log.decode('utf-8').strip().split('\n')
                for log in log_lines:
                    if not log.strip():
                        continue
                    ts, msg = log.strip().split(' ', maxsplit=1)
                    ts = dateutil.parser.parse(ts).replace(tzinfo=dateutil.tz.UTC)
                    msg = msg.strip()
                    logs.append(self.LogEntry(ts=ts, msg=msg, source=info))
                    info.last_ts = ts
            logs.sort()
            for entry in logs:
                self._add_log(entry.msg, save_path=entry.source.path, print_to_stdout=self._stdout)
            logs.clear()
            await asyncio.sleep(self._check_freq)


@dataclass(frozen=True)
class BuildSpec:
    image_tag: str
    display_name: str
    launcher_src: Path
    start_script_src: Path
    params_filename: str
    runtime_objects: tuple[Any, ...]
    extra_runtime_sources: tuple[Path, ...]


class Orchestrator:
    """Orchestrator class that handles MHAgentA execution.

    Orchestrator handles definition of agents and their consequent containerization and deployment. It also allows you
    to define default parameters shared by all the agents handles by it (can be overridden by individual agents)

    """
    SAVE_SUBDIR = 'out'
    LOG_CHECK_FREQ = 1.

    TRACE: int = 5
    DEBUG: int = 10
    PROGRESS: int = 15
    INFO: int = 20
    WARNING: int = 30
    ERROR: int = 40
    CRITICAL: int = 50

    def __init__(
            self,
            save_dir: str | os.PathLike,
            port_mapping: dict[int, int] | None = None,
            step_frequency: float = 1.,
            status_frequency: float = 5.,
            control_frequency: float = -1.,
            exec_start_time: float | None = None,
            agent_start_delay: float = 60.,
            exec_duration: float = 60.,
            save_format: Literal['json', 'dill'] = 'json',
            resume: bool = False,
            log_level: int = logging.INFO,
            log_format: str | None = None,
            status_msg_format: str = '[status_upd]::{}',
            connector_cls: type[Connector] = RabbitMQConnector,
            connector_kwargs: dict[str, Any] | None = None,
            mas_rmq_uri: str | Literal['default'] | None = None,
            mas_rmq_close_on_exit: bool = True,
            mas_rmq_exchange_name: str | None = None,
            save_logs: bool = True,
            no_stdout_logs: bool = False
    ) -> None:
        """
        Constructor method for Orchestrator.

        Args:
            save_dir (str | os.PathLike): Root directory for storing agents' states, logs, and temporary files.
            port_mapping (dict[int, int], optional): Mapping between internal docker container ports and host ports.
            step_frequency (float, optional, default=1.0): For agent modules with periodic step functions, the
                frequency in seconds of the step function calls that modules will try to maintain (unless their
                execution takes longer, then the next iteration will be scheduled without a time delay).
            status_frequency (float, optional, default=10.0): Frequency with which agent modules will report their
                statuses to the agent's root controller (error statuses will be reported immediately, regardless of
                the value).
            control_frequency (float, optional): Frequency of agent modules' internal clock when there's no tasks
                pending. If undefined or not positive, there will be no scheduling delay.
            exec_start_time (float, optional): Unix timestamp in seconds of when the agent's execution will try to
                start (unless agent's initialization takes longer than that; in this case the agent will start
                execution as soon as it finishes initializing). If not specified, agents will start execution
                immediately after their initialization.
            agent_start_delay (float, optional, default=60.0): Delay in seconds before agents starts execution. Use when
                `exec_start_time` is not defined to stage synchronous agents start at `agent_start_delay` seconds from
                the `run()` or `arun()` call.
            exec_duration (float, optional, default=60.0):  Time limit for agent execution in seconds. All agents will
                time out after this time.
            save_format (Literal['json', 'dill'], optional, default='json'): Format of agent modules state save files. JSON
                is more restrictive of what fields the states can include, but it is readable by humans.
            resume (bool, optional, default=False): Specifies whether to use save module states when restarting an
                agent with preexisting ID.
            log_level (int, optional, default=logging.INFO): Logging level.
            log_format (str, optional): Format of agent log messages. Defaults to
                `[%(agent_time)f|%(mod_time)f|%(exec_time)s][%(levelname)s]::%(tags)s::%(message)s`
            status_msg_format (str, optional): Format of agent status messages for external monitoring. Defaults to
                `[status_upd]::{}`
            connector_cls (type[Connector], optional, default=RabbitMQConnector): internal connector class that
                implements communication between modules. MHAgentA agents use RabbitMQ-based connectors by default.
            connector_kwargs (dict[str, Any], optional): Additional keyword arguments for connector. For
                RabbitMQConnector, the default parameters are: {`host`: 'localhost', `port`: 5672, `prefetch_count`: 1}.
            mas_rmq_uri (str, optional): URI of RabbitMQ server for multi-agent communication. Will try to start
                a RabbitMQ docker server at localhost:5672 if 'default'.
            mas_rmq_close_on_exit (bool, optional, default=True): Whether to close RabbitMQ server when exiting.
            mas_rmq_exchange_name (str, optional): Name of RabbitMQ exchange for inter-agent communication.
                Defaults to 'mhagenta'.
            save_logs (bool, optional, default=True): Whether to save agent logs. If True, saves each agent's logs to
                `<agent_id>.log` at the root of the `save_dir`. Defaults to True.
            no_stdout_logs (bool, optional, default=False): Whether to suppress stdout logs. Defaults to False.
        """
        if os.name != 'nt' and os.name != 'posix':
            raise RuntimeError(f'OS {os.name} is not supported.')

        self._agents: dict[str, AgentEntry] = dict()
        self._environments: dict[str, EnvironmentEntry] = dict()

        save_dir = Path(save_dir).resolve()
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir = save_dir

        assert mhagenta.__file__ is not None, 'Cannot locate mhagenta package!'
        self._package_dir = str(Path(mhagenta.__file__).parent.resolve())

        self._connector_cls = connector_cls if connector_cls else RabbitMQConnector
        if connector_kwargs is None and connector_cls == RabbitMQConnector:
            self._connector_kwargs = {
                'host': 'localhost',
                'port': 5672,
                'prefetch_count': 1
            }
        else:
            self._connector_kwargs = connector_kwargs

        self._port_mapping = port_mapping if port_mapping else {}

        self._step_frequency = step_frequency
        self._status_frequency = status_frequency
        self._control_frequency = control_frequency
        self._exec_start_time = exec_start_time
        self._exec_duration_sec = exec_duration
        self._agent_start_delay = agent_start_delay

        self._save_format = save_format
        self._resume = resume

        self._log_level = log_level
        self._log_format = log_format if log_format else DEFAULT_LOG_FORMAT
        self._status_msg_format = status_msg_format

        self._save_logs = save_logs

        self._mas_rmq_uri = mas_rmq_uri if mas_rmq_uri != 'default' else 'localhost:5672'
        self._mas_rmq_uri_internal = mas_rmq_uri if mas_rmq_uri != 'default' else 'localhost:5672'
        if self._mas_rmq_uri_internal is not None and 'localhost' in self._mas_rmq_uri_internal:
            self._mas_rmq_uri_internal = self._mas_rmq_uri_internal.replace('localhost', EDirectory.localhost_linux if sys.platform == 'linux' else EDirectory.localhost_win)
        self._mas_rmq_close_on_exit = mas_rmq_close_on_exit
        self._mas_rmq_container: Container | None = None
        self._mas_rmq_exchange_name = mas_rmq_exchange_name

        self._start_time: float = -1.
        self._simulation_end_ts = -1.

        self._docker_client: docker.DockerClient = docker.from_env()
        self._rabbitmq_image: Image | None = None
        self._base_image: Image | None = None

        self._task_group: TaskGroup | None = None
        self._force_run = False

        self._monitor: Monitor | None = None

        self._running = False
        self._stopping = False
        self._all_stopped = False

        self._log_parser = LogParser(
            stop_checker=lambda: self._stopping and self._containers_stopped,
            check_freq=1.,
            save_logs=self._save_dir if save_logs else None,
            no_stdout=no_stdout_logs
        )

    def add_environment(
            self,
            base: MHAEnvBase,
            env_id: str = 'environment',
            host: str | None = 'localhost',
            port: int | None = 5672,
            exec_duration: float | None = None,
            exchange_name: str | None = None,
            init_script: os.PathLike | str | None = None,
            requirements_path: os.PathLike | str | None = None,
            port_mapping: dict[int, int] | None = None,
            log_tags: list[str] | None = None,
            log_level: int | str | None = None,
            log_format: str | None = None,
            tags: Iterable[str] | None = None,
            extra_runtime_sources: os.PathLike | str | Iterable[os.PathLike | str] | None = None
    ) -> None:
        """
        Add a configuration of an environment to build at the runtime.

        Args:
            base (MHAEnvBase): The base environment object implementing the environment behaviour.
            env_id (str): Unique identifier for the environment. Defaults to 'environment'.
            host (str, optional): RabbitMQ host address; will use one from the Orchestrator if None. Defaults to 'localhost'.
            port (int, optional): RabbitMQ port; will use one from the Orchestrator if None. Defaults to 5672.
            exec_duration (float, optional): execution duration of the environment, will derive from the Orchestrator
                configuration if None. Defaults to None.
            exchange_name (str, optional): Name of RabbitMQ exchange for inter-agent communication. Will use the default
                one for MAS if None. Defaults to None.
            init_script (os.PathLike | str, optional): Path to an optional bash script to be run before launching the environment.
                Use it to install additional non-Python dependencies. Defaults to None.
            requirements_path (os.PathLike | str, optional): Path to Python dependencies file. Defaults to None.
            port_mapping (dict[int, int], optional): Mapping between internal docker container ports and host ports.
                Defaults to the Orchestrator's `port_mapping`.
            log_tags (list[str], optional): List of tags to add to log messages. Defaults to None.
            log_level (int, optional): Log level; will use the Orchestrator's log level in None. Defaults to None.
            log_format (str, optional): Log format string; will use the Orchestrator's log format. Defaults to None.
            tags (Iterable[str], optional): List of tags for agent directory. Defaults to None.
            extra_runtime_sources (os.PathLike | str | Iterable[os.PathLike | str], optional): Additional *local*
                runtime sources to be added to the environment's execution context. Use `requirements_path` to add
                third-party modules. Defaults to None.

        Returns:

        """
        from mhagenta.defaults.communication.rabbitmq import RMQEnvironment
        if host is None:
            if self._mas_rmq_uri is None:
                raise ValueError('No RabbitMQ URI specified for the environment!')
            mas_host, mas_port = self._mas_rmq_uri.split(':')
            host = mas_host
            if port is None:
                port = mas_port
        env_dir = self._save_dir.resolve() / env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        if tags is not None:
            tags = list(tags)
        exec_duration = exec_duration if exec_duration else self._exec_duration_sec
        if self._agent_start_delay is not None and self._agent_start_delay > 0:
            exec_duration += self._agent_start_delay
        elif self._exec_start_time is not None and self._exec_start_time > 0:
            exec_duration += (self._exec_start_time - time.time())
        kwargs = {
            'env_class': RMQEnvironment,
            'base': base,
            'env_id': env_id,
            'host': host if host != 'localhost' and host != '127.0.0.1' else 'host.docker.internal',
            'port': port,
            'exec_duration': exec_duration,
            'exchange_name': exchange_name if exchange_name is not None else self._mas_rmq_exchange_name,
            'start_time_reference': None,
            'save_dir': f'/{self.SAVE_SUBDIR}',
            'save_format': self._save_format,
            'log_id': env_id,
            'log_tags': log_tags if log_tags is not None else [],
            'log_format': log_format if log_format is not None else self._log_format,
            'log_level': log_level if log_level is not None else self._log_level,
            'tags': tags
        }

        self._environments[env_id] = EnvironmentEntry(
            env_id=env_id,
            kwargs=kwargs,
            address={
                'exchange_name': kwargs['exchange_name'],
                'env_id': env_id
            },
            dir=env_dir,
            tags=tags,
            port_mapping=port_mapping if port_mapping else self._port_mapping,
            script_path=Path(init_script).resolve() if init_script is not None else None,
            requirements_path=Path(requirements_path).resolve() if requirements_path is not None else None,
            extra_runtime_sources=self._normalize_runtime_sources(extra_runtime_sources)
        )

    @staticmethod
    def _update_external_host(module: ActuatorBase | PerceptorBase):
        if 'external'not  in module.tags:
            return
        conn_params = getattr(module, 'conn_params', None)
        if conn_params is None:
            warnings.warn(f'Module {module} has "external" tag but no connection parameters, ignoring it.')
            return
        if conn_params['host'] == 'localhost':
            conn_params['host'] = EDirectory.localhost_linux if sys.platform == 'linux' else EDirectory.localhost_win

    @staticmethod
    def _try_extend_ids(modules: Iterable[ModuleBase] | None, module_ids: set[str]) -> None:
        if modules is None:
            return
        for module in modules:
            if module.module_id in module_ids:
                raise ValueError(f'Found a duplicate module ID {module.module_id}!')
            module_ids.add(module.module_id)

    def add_agent(
            self,
            agent_id: str,
            perceptors: Iterable[PerceptorBase] | PerceptorBase,
            actuators: Iterable[ActuatorBase] | ActuatorBase,
            ll_reasoners: Iterable[LLReasonerBase] | LLReasonerBase,
            learners: Iterable[LearnerBase] | LearnerBase | None = None,
            knowledge: Iterable[KnowledgeBase] | KnowledgeBase | None = None,
            hl_reasoners: Iterable[HLReasonerBase] | HLReasonerBase | None = None,
            goal_graphs: Iterable[GoalGraphBase] | GoalGraphBase | None = None,
            memory: Iterable[MemoryBase] | MemoryBase | None = None,
            num_copies: int = 1,
            step_frequency: float | None = None,
            status_frequency: float | None = None,
            control_frequency: float | None = None,
            exec_start_time: float | None = None,
            start_delay: float = 0.,
            exec_duration: float | None = None,
            resume: bool | None = None,
            init_script: os.PathLike | str | None = None,
            requirements_path: os.PathLike | str | None = None,
            log_level: int | None = None,
            port_mapping: dict[int, int] | None = None,
            connector_cls: type[Connector] | None = None,
            connector_kwargs: dict[str, Any] | None = None,
            tags: Iterable[str] | None = None,
            extra_runtime_sources: os.PathLike | str | Iterable[os.PathLike | str] | None = None
    ) -> None:
        """Define an agent model to be added to the execution.

        This can be either a single agent, a set of identical agents following the same structure model.

        Args:
            agent_id (str): A unique identifier for the agent.
            perceptors (Iterable[PerceptorBase] | PerceptorBase): Definition(s) of agent's perceptor(s).
            actuators (Iterable[ActuatorBase] | ActuatorBase): Definition(s) of agent's actuator(s).
            ll_reasoners (Iterable[LLReasonerBase] | LLReasonerBase): Definition(s) of agent's ll_reasoner(s).
            learners (Iterable[LearnerBase] | LearnerBase, optional): Definition(s) of agent's learner(s).
            knowledge (Iterable[KnowledgeBase] | KnowledgeBase, optional): Definition(s) of agent's knowledge model(s).
            hl_reasoners (Iterable[HLReasonerBase] | HLReasonerBase, optional): Definition(s) of agent's hl_reasoner(s).
            goal_graphs (Iterable[GoalGraphBase] | GoalGraphBase, optional): Definition(s) of agent's goal_graph(s).
            memory (Iterable[MemoryBase] | MemoryBase, optional): Definition(s) of agent's memory structure(s).
            num_copies (int, optional, default=1): Number of copies of the agent to instantiate at runtime.
            step_frequency (float, optional): For agent modules with periodic step functions, the frequency in seconds
                of the step function calls that modules will try to maintain (unless their execution takes longer, then
                the next iteration will be scheduled without a time delay). Defaults to the Orchestrator's
                `step_frequency`.
            status_frequency (float, optional): Frequency with which agent modules will report their statuses to the
                agent's root controller (error statuses will be reported immediately, regardless of the value).
                Defaults to the Orchestrator's `status_frequency`.
            control_frequency (float, optional): Frequency of agent modules' internal clock when there's no tasks
                pending. If undefined or not positive, there will be no scheduling delay. Defaults to the
                Orchestrator's `control_frequency`.
            exec_start_time (float, optional): Unix timestamp in seconds of when the agent's execution will try to
                start (unless agent's initialization takes longer than that; in this case the agent will start
                execution as soon as it finishes initializing). Defaults to the Orchestrator's `exec_start_time`.
            start_delay (float, optional, default=0.0): A time offset from the global execution time start when this agent will
                attempt to start its own execution.
            exec_duration (float, optional): Time limit for agent execution in seconds. The agent will time out after
                this time. Defaults to the Orchestrator's `exec_duration`.
            resume (bool, optional): Specifies whether to use save module states when restarting an agent with
                preexisting ID. Defaults to the Orchestrator's `resume`.
            init_script (os.PathLike | str, optional): Path to an optional bash script to be run before launching the agent.
                Use it to install additional non-Python dependencies. Defaults to None.
            requirements_path (os.PathLike | str, optional): Additional Python requirements to install on agent side.
            log_level (int, optional):  Logging level for the agent. Defaults to the Orchestrator's `log_level`.
            port_mapping (dict[int, int], optional): Mapping between internal docker container ports and host ports.
                Defaults to the Orchestrator's `port_mapping`.
            connector_cls (type[Connector], optional): internal connector class that implements communication between
                modules. Defaults to the Orchestrator's `connector_cls`.
            connector_kwargs (dict[str, Any], optional): Additional keyword arguments for connector. Defaults to
                the Orchestrator's `connector_kwargs`.
            tags (Iterable[str], optional): a list of tags associated with this agent for directory search.
            extra_runtime_sources (os.PathLike | str | Iterable[os.PathLike | str], optional): Additional *local*
                runtime sources to be added to the agent's execution context. Use `requirements_path` to add
                third-party modules. Defaults to None.

        """
        if agent_id in self._agents:
            raise KeyError(f'Agent with ID "{agent_id}" already exists!')

        module_ids: set[str] = set()

        if isinstance(perceptors, PerceptorBase):
            perceptors = [perceptors]
        if isinstance(actuators, ActuatorBase):
            actuators = [actuators]
        if isinstance(ll_reasoners, LLReasonerBase):
            ll_reasoners = [ll_reasoners]
        if isinstance(knowledge, KnowledgeBase):
            knowledge = [knowledge]
        if isinstance(hl_reasoners, HLReasonerBase):
            hl_reasoners = [hl_reasoners]
        if isinstance(goal_graphs, GoalGraphBase):
            goal_graphs = [goal_graphs]
        if isinstance(memory, MemoryBase):
            memory = [memory]
        if isinstance(learners, LearnerBase):
            learners = [learners]

        for perceptor in perceptors:
            if perceptor.module_id in module_ids:
                raise ValueError(f'Found duplicate module ID {perceptor.module_id}!')
            module_ids.add(perceptor.module_id)
            self._update_external_host(perceptor)

        for actuator in actuators:
            if actuator.module_id in module_ids:
                raise ValueError(f'Found duplicate module ID {actuator.module_id}!')
            module_ids.add(actuator.module_id)
            self._update_external_host(actuator)

        self._try_extend_ids(ll_reasoners, module_ids)
        self._try_extend_ids(knowledge, module_ids)
        self._try_extend_ids(hl_reasoners, module_ids)
        self._try_extend_ids(goal_graphs, module_ids)
        self._try_extend_ids(memory, module_ids)
        self._try_extend_ids(learners, module_ids)

        kwargs = {
            'agent_id': agent_id,
            'connector_cls': connector_cls if connector_cls else self._connector_cls,
            'perceptors': perceptors,
            'actuators': actuators,
            'll_reasoners': ll_reasoners,
            'learners': learners,
            'knowledge': knowledge,
            'hl_reasoners': hl_reasoners,
            'goal_graphs': goal_graphs,
            'memory': memory,
            'connector_kwargs': connector_kwargs if connector_kwargs else self._connector_kwargs,
            'step_frequency': self._step_frequency if step_frequency is None else step_frequency,
            'status_frequency': self._status_frequency if status_frequency is None else status_frequency,
            'control_frequency': self._control_frequency if control_frequency is None else control_frequency,
            'exec_start_time': self._exec_start_time if exec_start_time is None else exec_start_time,
            'start_delay': start_delay,
            'exec_duration': self._exec_duration_sec if exec_duration is None else exec_duration,
            'save_dir': f'/{self.SAVE_SUBDIR}',
            'save_format': self._save_format,
            'resume': self._resume if resume is None else resume,
            'log_level': self._log_level if log_level is None else log_level,
            'log_format': self._log_format,
            'status_msg_format': self._status_msg_format
        }

        self._agents[agent_id] = AgentEntry(
            agent_id=agent_id,
            port_mapping=port_mapping if port_mapping else self._port_mapping,
            num_copies=num_copies,
            kwargs=kwargs,
            tags=tags,
            script_path=Path(init_script).resolve() if init_script is not None else None,
            requirements_path=Path(requirements_path).resolve() if requirements_path is not None else None,
            extra_runtime_sources=self._normalize_runtime_sources(extra_runtime_sources)
        )
        if self._task_group is not None:
            self._task_group.create_task(self._run_agent(self._agents[agent_id], force_run=self._force_run))

    def _compose_directory(self) -> Directory:
        directory = Directory()
        if self._mas_rmq_uri_internal is None:
            return directory

        host, port = self._mas_rmq_uri_internal.split(':')

        for env in self._environments.values():
            directory.external.add_env(
                env_id=env.env_id,
                address={
                    'exchange_name': env.kwargs['exchange_name'],
                    'env_id': env.env_id,
                    'host': host,
                    'port': port
                },
                tags=env.kwargs['tags']
            )

        for agent in self._agents.values():
            if agent.num_copies == 1:
                agent_ids = [agent.agent_id]
            else:
                agent_ids = [f'{agent.agent_id}_{i}' for i in range(agent.num_copies)]
            for agent_id in agent_ids:
                directory.external.add_agent(
                    agent_id=agent_id,
                    address={
                        'exchange_name': self._mas_rmq_exchange_name,
                        'agent_id': agent_id,
                        'host': host,
                        'port': port
                    },
                    tags=agent.tags
                )
        return directory

    def _docker_build_base(
            self,
            mhagenta_version: str = 'latest',
            local_build: os.PathLike | str | None = None,
            prerelease: bool = False
    ) -> None:
        if not mhagenta_version:
            mhagenta_version = CONTAINER_VERSION

        if self._base_image is None:
            print(f'===== LOOKING FOR AGENT BASE IMAGE: {REPO}:{mhagenta_version} =====')
            try:
                self._base_image = self._docker_client.images.list(name=f'{REPO}:{mhagenta_version}')[0]
            except IndexError:
                print('\tIMAGE NOT FOUND LOCALLY...')
                if local_build is None:
                    print(f'===== PULLING AGENT BASE IMAGE: {REPO}:{mhagenta_version} =====')
                    try:
                        self._base_image = self._docker_client.images.pull(REPO, mhagenta_version)
                        print('\tSUCCESSFULLY PULLED THE IMAGE!')
                        return
                    except docker.errors.ImageNotFound:
                        print('\tPULLING AGENT BASE IMAGE FAILED...')
                build_dir = self._save_dir.resolve() / 'tmp' / 'mha-base'
                try:
                    try:
                        print(f'===== PULLING RABBITMQ BASE IMAGE: {REPO}:rmq =====')
                        self._docker_client.images.pull(REPO, tag='rmq')
                    except docker.errors.ImageNotFound:
                        print('Pulling failed...')
                        print(f'===== BUILDING RABBITMQ BASE IMAGE: {REPO}:rmq =====')
                        if self._rabbitmq_image is None:
                            self._rabbitmq_image, _ = (
                                self._docker_client.images.build(path=RABBIT_IMG_PATH,
                                                                 tag=f'{REPO}:rmq',
                                                                 rm=True,
                                                                 quiet=False
                                                                 ))
                    print(f'===== BUILDING AGENT BASE IMAGE: {REPO}:{mhagenta_version} =====')
                    shutil.copytree(BASE_IMG_PATH, build_dir, dirs_exist_ok=True)
                    if local_build is not None:
                        local_build = Path(local_build).resolve()
                        shutil.copytree(local_build / 'mhagenta', build_dir / 'mha-local' / 'mhagenta')
                        shutil.copy(local_build / 'pyproject.toml', build_dir / 'mha-local' / 'pyproject.toml')
                        shutil.copy(local_build / 'README.md', build_dir / 'mha-local' / 'README.md')
                    else:
                        (build_dir / 'mha-local').mkdir(parents=True, exist_ok=True)

                    self._base_image, _ = (
                        self._docker_client.images.build(
                            path=str(build_dir),
                            buildargs={
                                'SRC_IMAGE': REPO,
                                'SRC_TAG': 'rmq',
                                'PRE_VERSION': 'true' if prerelease else 'false',
                                'LOCAL': 'false' if local_build is None else 'true',
                            },
                            tag=f'{REPO}:{mhagenta_version}',
                            rm=True,
                            quiet=False
                        ))
                except Exception as ex:
                    shutil.rmtree(build_dir, ignore_errors=True)
                    raise ex
                shutil.rmtree(build_dir)

    def _logged_build(self, *args, **kwargs) -> Any:
        try:
            results = self._docker_client.images.build(*args, **kwargs)
            return results
        except docker.errors.BuildError as e:
            print('Build error encountered!')
            for log in e.build_log:
                if 'stream' in log:
                    msg = log['stream'].strip()
                    if msg:
                        print(f'[stream] {msg}')
                elif 'error' in log:
                    print(f'[error ] {log['error']}')
                    if 'errorDetail' in log:
                        for d_key, d_val in log['errorDetail'].items():
                            print(f'\t[{d_key}] {d_val}')
            raise e

    @staticmethod
    def _copy_extras(entry: Entry, build_dir: Path) -> None:
        if entry.script_path:
            shutil.copy(entry.script_path, (build_dir / 'src' / 'init_script.sh').resolve())

        if entry.requirements_path:
            shutil.copy(entry.requirements_path, (build_dir / 'src' / 'requirements.txt').resolve())

    def _docker_build(
            self,
            build_dir: Path,
            src_tag: tuple[str, str],
            tag: str,
    ) -> Image:
        image, _ = self._logged_build(
            path=str(build_dir.resolve()),
            buildargs={
                'SRC_IMAGE': src_tag[0],
                'SRC_VERSION': src_tag[1]
            },
            tag=tag,
            rm=True,
            quiet=False
        )
        shutil.rmtree(build_dir)
        return image

    def _docker_build_runtime(
            self,
            spec: BuildSpec,
            out_dir: Path,
            params: dict[str, Any],
            entry: Entry,
            rebuild_image: bool
    ) -> Image:
        try:
            img = self._docker_client.images.list(name=spec.image_tag)[0]
            if rebuild_image:
                img.remove(force=True)
            else:
                print(f'===== {spec.display_name} IMAGE FOUND: {spec.image_tag} (NO REBUILD REQUESTED) =====')
                return img
        except IndexError:
            if not rebuild_image:
                raise ValueError(f'Image {spec.image_tag} is not found!')

        assert self._base_image is not None, 'Base image is not set!'
        base_image_tags = cast(list[str], self._base_image.tags)
        if len(base_image_tags) > 1:
            base_image_tags[:] = [t for t in base_image_tags if not t.endswith('latest')]
        base_image_tag = sorted(base_image_tags, key=lambda t: len(t))[-1]
        print(f'===== BUILDING {spec.display_name} IMAGE: {spec.image_tag} FROM {base_image_tag} =====')

        if self._force_run and out_dir.exists():
            shutil.rmtree(out_dir)

        (out_dir / self.SAVE_SUBDIR).mkdir(parents=True)
        build_dir = out_dir / 'tmp/'

        shutil.copytree(AGENT_IMG_PATH, build_dir.resolve())
        src_dir = (build_dir / 'src')
        src_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(spec.launcher_src, src_dir / spec.launcher_src.name)
        shutil.copy(spec.start_script_src, src_dir / 'start.sh')

        self._copy_extras(entry, build_dir)

        runtime_sources: dict[str, Path] = {}

        self._copy_runtime_python_modules(spec.runtime_objects, src_dir, runtime_sources)
        self._copy_explicit_runtime_sources(spec.extra_runtime_sources, src_dir, runtime_sources)

        with open((src_dir / spec.params_filename).resolve(), 'wb') as f:
            dill.dump(params, f, recurse=True)

        base_tag_split = cast(tuple[str, str], tuple(base_image_tag.split(':', 1)))
        return self._docker_build(build_dir=build_dir,
                                  src_tag=base_tag_split,
                                  tag=spec.image_tag
                                  )

    def _docker_build_agent(
            self,
            agent: AgentEntry,
            rebuild_image: bool = True,
    ) -> None:
        assert mhagenta.__file__ is not None, 'Cannot locate mhagenta package!'
        assert mhagenta.core.__file__ is not None, 'Cannot locate mhagenta.core package!'

        agent_dir = self._save_dir.resolve() / agent.agent_id
        agent.dir = agent_dir
        agent.save_dir = agent_dir / self.SAVE_SUBDIR

        params = agent.kwargs.copy()
        params['directory'] = self._compose_directory()
        exec_start_time = self._start_time if params['exec_start_time'] is None else params['exec_start_time']
        params['exec_start_time'] = exec_start_time + self._agent_start_delay

        end_estimate = params['exec_start_time'] + params['start_delay'] + params['exec_duration']
        self._simulation_end_ts = max(self._simulation_end_ts, end_estimate)

        spec = BuildSpec(
            image_tag=f'mhagent:{agent.agent_id}',
            display_name='AGENT',
            launcher_src=Path(mhagenta.core.__file__).parent.resolve() / 'agent_launcher.py',
            start_script_src=Path(mhagenta.__file__).parent.resolve() / 'scripts' / 'start.sh',
            params_filename='agent_params',
            runtime_objects=tuple(agent.modules),
            extra_runtime_sources=agent.extra_runtime_sources
        )

        agent.image = self._docker_build_runtime(
            spec=spec,
            out_dir=agent_dir,
            params=params,
            entry=agent,
            rebuild_image=rebuild_image
        )

    def _docker_build_env(
            self,
            environment: EnvironmentEntry,
            rebuild_image: bool = True,
    ) -> None:
        assert mhagenta.__file__ is not None, 'Cannot locate magenta package!'
        assert mhagenta.environment.__file__ is not None, 'Cannot locate magenta.environment package!'

        env_dir = self._save_dir.resolve() / environment.env_id
        environment.dir = env_dir
        params = environment.kwargs.copy()
        params['exec_duration'] += (self._start_time - time.time())

        spec = BuildSpec(
            image_tag=f'mhagent-env:{environment.env_id}',
            display_name='ENVIRONMENT',
            launcher_src=Path(mhagenta.environment.__file__).parent.resolve() / 'environment_launcher.py',
            start_script_src=Path(mhagenta.__file__).parent.resolve() / 'scripts' / 'env_start.sh',
            params_filename='env_params',
            runtime_objects=(environment.base,),
            extra_runtime_sources=environment.extra_runtime_sources
        )

        environment.image = self._docker_build_runtime(
            spec=spec,
            out_dir=env_dir,
            params=params,
            entry=environment,
            rebuild_image=rebuild_image
        )

    async def _run_agent(
            self,
            agent: AgentEntry,
            force_run: bool = False
    ) -> None:
        if agent.num_copies == 1:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS CONTAINER \"{agent.agent_id}\" =====')
        else:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS '
                  f'{agent.num_copies} CONTAINERS \"{agent.agent_id}_#\" =====')
        agent.containers = dict()
        for i in range(agent.num_copies):
            if agent.num_copies == 1:
                agent_name = agent.agent_id
                agent_dir = (agent.dir / self.SAVE_SUBDIR).resolve()
            else:
                agent_name = f'{agent.agent_id}_{i}'
                agent_dir = (agent.dir.with_name(agent_name) / self.SAVE_SUBDIR).resolve()

            agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                container = self._docker_client.containers.get(agent_name)
                if force_run:
                    container.remove(force=True)
                else:
                    raise NameError(f'Container {agent_name} already exists')
            except NotFound:
                pass

            if self._mas_rmq_uri_internal is not None:
                host, port = self._mas_rmq_uri_internal.split(':')
                port = int(port) + 10_000
            else:
                host, port = None, None

            assert agent.containers is not None
            agent.containers[agent_name] = self._docker_client.containers.run(
                image=agent.image,
                detach=True,
                name=agent_name,
                environment={
                    'AGENT_ID': agent_name,
                    'DOCKER_NAME': agent_name,
                    'RMQ_HOST': host,
                    'RMQ_PORT': port,
                    'VERBOSE': "true" if self._log_level <= self.PROGRESS else "false"
                },
                volumes={
                    str(agent_dir): {'bind': f'/{self.SAVE_SUBDIR}', 'mode': 'rw'}
                },
                extra_hosts={'host.docker.internal': 'host-gateway'},
                ports=agent.port_mapping
            )
        self._log_parser.add_container(agent)

    async def _run_env(
            self,
            environment: EnvironmentEntry,
            force_run: bool = False
    ) -> None:
        print(f'===== RUNNING ENVIRONMENT IMAGE \"mhagent-env:{environment.env_id}\" AS CONTAINER \"{environment.env_id}\" =====')

        env_dir = (environment.dir / self.SAVE_SUBDIR).resolve()
        env_dir.mkdir(parents=True, exist_ok=True)

        try:
            container = self._docker_client.containers.get(environment.env_id)
            if force_run:
                container.remove(force=True)
            else:
                raise NameError(f'Container {environment.env_id} already exists')
        except NotFound:
            pass

        if self._mas_rmq_uri_internal is not None:
            host, port = self._mas_rmq_uri_internal.split(':')
            port = int(port) + 10_000
        else:
            host, port = None, None
        environment.containers = {environment.env_id: self._docker_client.containers.run(
            image=environment.image,
            detach=True,
            name=environment.env_id,
            environment={
                'AGENT_ID': '',
                'DOCKER_NAME': environment.env_id,
                'RMQ_HOST': host,
                'RMQ_PORT': port,
                'VERBOSE': "true" if self._log_level <= self.PROGRESS else "false"
            },
            volumes={
                str(env_dir): {'bind': f'/{self.SAVE_SUBDIR}', 'mode': 'rw'}
            },
            extra_hosts={'host.docker.internal': 'host-gateway'},
            ports=environment.port_mapping
        )}
        self._log_parser.add_container(environment)

    async def arun(
            self,
            mhagenta_version: str = 'latest',
            force_run: bool = False,
            gui: bool = False,
            rebuild_agents: bool = True,
            rebuild_envs: bool = True,
            local_build: os.PathLike | str | None = None,
            prerelease: bool = False,
            keep_containers: bool = False
    ) -> None:
        """Run all the agents as an async method. Use in case you want to control the async task loop yourself.

        Args:
            mhagenta_version (str, optional): Version of mhagenta base container. Defaults to 'latest'.
            force_run (bool, optional, default=False): In case containers with some of the specified agent IDs exist,
                specify whether to force remove the old container to run the new ones. Otherwise, an exception will be
                raised.
            gui (bool, optional, default=False): Specifies whether to open the log monitoring window for the
                orchestrator.
            rebuild_agents (bool, optional, default=True): Whether to rebuild the agent containers. Defaults to True.
            rebuild_envs (bool, optional, default=True): Whether to rebuild the environment containers. Defaults to True.
            local_build (os.PathLike | str, optional): Specifies the path to a local build of MHAgentA (as opposed to the latest
                one from PyPI) to be used for building agents.
            prerelease (bool, optional, default=False): Specifies whether to allow agents to use the latest prerelease
                version of mhagenta while building the container.
            keep_containers (bool, optional, default=False): Whether to keep or remove the agent and environment
                containers after the execution.

        Raises:
            NameError: Raised if a container for one of the specified agent IDs already exists and `force_run` is False.

        """

        self._start_time = time.time()

        # print(f'[Orchestrator] Starting execution at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        if self._base_image is None:
            self._docker_build_base(mhagenta_version=mhagenta_version, local_build=local_build, prerelease=prerelease)

        self._force_run = force_run
        for env in self._environments.values():
            self._docker_build_env(env, rebuild_image=rebuild_envs)
        for agent in self._agents.values():
            self._docker_build_agent(agent, rebuild_image=rebuild_agents)

        if gui:
            self._monitor = Monitor()

        self._running = True

        self.start_rabbitmq()

        async with asyncio.TaskGroup() as tg:
            self._task_group = tg
            if gui:
                tg.create_task(self._monitor.run())
            # if self._environment is not None:
            #     tg.create_task(self._read_logs())
            tg.create_task(self._simulation_end_timer())
            for env in self._environments.values():
                tg.create_task(self._run_env(env, force_run=force_run))
                # tg.create_task(self._read_logs(env, gui=False))
                # self._log_parser.add_container(env)
            for agent in self._agents.values():
                tg.create_task(self._run_agent(agent, force_run=force_run))
                # tg.create_task(self._read_logs(agent, gui))
                # self._log_parser.add_container(agent)
            tg.create_task(self._log_parser.run())
        self._running = False
        for agent in self._agents.values():
            assert agent.containers is not None
            for container in agent.containers.values():
                container.stop()
                if not keep_containers:
                    container.remove()
        for env in self._environments.values():
            assert env.containers is not None
            env.containers[env.env_id].stop()
            if not keep_containers:
                env.containers[env.env_id].remove()
        if self._mas_rmq_container is not None and self._mas_rmq_close_on_exit:
            try:
                self._mas_rmq_container.stop()
            except Exception:
                pass
        print('===== EXECUTION FINISHED =====')

    def run(
            self,
            mhagenta_version='latest',
            force_run: bool = False,
            gui: bool = False,
            rebuild_agents: bool = True,
            rebuild_envs: bool = True,
            local_build: os.PathLike | str | None = None,
            prerelease: bool = False,
            keep_containers: bool = False
    ) -> None:
        """Run all the agents.

        Args:
            mhagenta_version (str, optional): Version of mhagenta base container. Defaults to 'latest'.
            force_run (bool, optional, default=False): In case containers with some of the specified agent IDs exist,
                specify whether to force remove the old container to run the new ones. Otherwise, an exception will be
                raised.
            gui (bool, optional, default=False): Specifies whether to open the log monitoring window for the
                orchestrator.
            rebuild_agents (bool, optional, default=True): Whether to rebuild the agents. Defaults to True.
            rebuild_envs (bool, optional, default=True): Whether to rebuild the environment containers. Defaults to True.
            local_build (os.PathLike | str, optional): Specifies the path to a local build of MHAgentA (as opposed to the latest
                one from PyPI) to be used for building agents.
            prerelease (bool, optional, default=False): Specifies whether to allow agents to use the latest prerelease
                version of mhagenta while building the container.
            keep_containers (bool, optional, default=False): Whether to keep or remove the agent and environment
                containers after the execution.

        Raises:
            NameError: Raised if a container for one of the specified agent IDs already exists and `force_run` is False.

        """
        asyncio.run(self.arun(
            mhagenta_version=mhagenta_version,
            force_run=force_run,
            gui=gui,
            rebuild_agents=rebuild_agents,
            rebuild_envs=rebuild_envs,
            local_build=local_build,
            prerelease=prerelease,
            keep_containers=keep_containers
        ))

    @staticmethod
    def _agent_stopped(agent: AgentEntry | EnvironmentEntry) -> bool:
        assert agent.containers is not None, 'Agent containers are not set!'
        for container in agent.containers.values():
            container.reload()
        return all([container.status == 'exited' for container in agent.containers.values()])

    @property
    def _agents_stopped(self) -> bool:
        if self._all_stopped:
            return True
        for agent in self._agents.values():
            if not self._agent_stopped(agent):
                return False
        return True

    @property
    def _containers_stopped(self) -> bool:
        if self._all_stopped:
            return True
        for agent in self._agents.values():
            if not self._agent_stopped(agent):
                return False
        for env in self._environments.values():
            if not self._agent_stopped(env):
                return False
        self._all_stopped = True
        return True

    async def _simulation_end_timer(self) -> None:
        # print(f'[Orchestrator] Simulation end time: {datetime.fromtimestamp(self._simulation_end_ts).strftime("%Y-%m-%d %H:%M:%S")} '
        #       f'(in {self._simulation_end_ts - time.time():.3f}) seconds).')
        await asyncio.sleep(self._simulation_end_ts - time.time())
        self._stopping = True

    def __getitem__(self, agent_id: str) -> AgentEntry:
        return self._agents[agent_id]

    def start_rabbitmq(self) -> None:
        self._connect_rabbitmq()

    def _connect_rabbitmq(self) -> None:
        if self._mas_rmq_uri_internal is None:
            return
        try:
            host, port = self._mas_rmq_uri.split(':') if ':' in self._mas_rmq_uri_internal else (self._mas_rmq_uri_internal, 5672)
            connection = BlockingConnection(pika.ConnectionParameters(host, port))
            connection.close()
        except AMQPConnectionError:
            self._mas_rmq_container = self._docker_client.containers.run(
                image=DEFAULT_RMQ_IMAGE,
                detach=True,
                name='mhagenta-rmq',
                ports={
                    '5672': 5672,
                    '15672': 15672
                },
                remove=True,
                tty=True
            )

    @staticmethod
    def _normalize_runtime_sources(sources: os.PathLike | str | Iterable[os.PathLike | str] | None) -> tuple[Path, ...]:
        if sources is None:
            return ()

        if isinstance(sources, (str, os.PathLike)):
            sources = (sources,)

        normalized: list[Path] = list()
        seen: set[Path] = set()

        for src in sources:
            path = Path(src).resolve()
            if not path.exists():
                raise FileNotFoundError(f'Runtime source path does not exist: "{path}"')
            if path.is_dir():
                if not path.name.isidentifier():
                    raise ValueError(f'Runtime package directory name is not importable: "{path.name}"')
                if not (path / '__init__.py').exists():
                    raise ValueError(f'Runtime package directory must be a package root with __init__.py: "{path}"')
            elif path.is_file():
                if path.suffix != '.py':
                    raise ValueError(f'Runtime source file must be a .py module: "{path}"')
                if path.name == '__init__.py':
                    raise ValueError(f'Pass the package directory instead of its __init__.py file: "{path}"')
                if not path.stem.isidentifier():
                    raise ValueError(f'Runtime package directory name is not importable: "{path.name}"')
                if (path.parent / '__init__.py').exists():
                    raise ValueError(f'Pass the package root, not a nested module: "{path}"')
            else:
                raise ValueError(f'Unsupported runtime source path: "{path}"')

            if path not in seen:
                normalized.append(path)
                seen.add(path)

        return tuple(normalized)

    @staticmethod
    def _module_source_root(module_name: str, module_file: Path) -> Path:
        root = module_file.parent
        steps_up = module_name.count('.')
        if module_file.name == '__init__.py':
            steps_up += 1
        for _ in range(steps_up):
            root = root.parent
        return root

    @staticmethod
    def _copy_runtime_sources(src: Path, dst_dir: Path, copied: dict[str, Path]) -> None:
        dst = dst_dir / src.name

        prev = copied.get(src.name)
        if prev is not None:
            if prev == src:
                return
            raise ValueError(f'Conflicting runtime sources targeting {src.name}: {prev} and {src}')

        if dst.exists():
            raise FileExistsError(f'Runtime source {src} would overwrite existing file: {dst}')

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        copied[src.name] = src

    @classmethod
    def _copy_explicit_runtime_sources(cls, sources: Iterable[Path], dst_dir: Path, copied: dict[str, Path]) -> None:
        for src in sources:
            cls._copy_runtime_sources(src, dst_dir, copied)

    @staticmethod
    def _copy_package_inits(root: Path, relative_parts: tuple[str, ...], dst_dir: Path) -> None:
        current_src = root
        current_dst = dst_dir

        for part in relative_parts:
            current_src = current_src / part
            current_dst = current_dst / part
            current_dst.mkdir(exist_ok=True)

            init_src = current_src / "__init__.py"
            init_dst = current_dst / "__init__.py"
            if init_src.exists() and not init_dst.exists():
                shutil.copy2(init_src, init_dst)

    @classmethod
    def _copy_runtime_package_subtree(
            cls,
            *,
            root: Path,
            package_parts: tuple[str, ...],
            dst_dir: Path,
            copied: dict[str, Path],
    ) -> None:
        src = root.joinpath(*package_parts)
        if not src.exists():
            raise FileNotFoundError(f"Runtime package source does not exist: {src}")

        target_key = ".".join(package_parts)
        prev = copied.get(target_key)
        if prev is not None:
            if prev == src:
                return
            raise ValueError(f"Conflicting runtime sources targeting {target_key}: {prev} and {src}")

        parent_parts = package_parts[:-1]
        cls._copy_package_inits(root, parent_parts, dst_dir)

        dst = dst_dir.joinpath(*package_parts)
        if dst.exists():
            raise FileExistsError(f"Runtime source {src} would overwrite existing file: {dst}")

        shutil.copytree(src, dst)
        copied[target_key] = src

    def _copy_runtime_python_modules(self, objects: Iterable[Any], dst_dir: Path, copied: dict[str, Path]) -> None:
        stdlib_dirs = {
            Path(p).resolve() for p in (sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')) if p
        }
        ignored_names = {'builtins', '__main__', '__mp_main__', 'mhagenta'}
        venv_parts = {'site-packages', 'dist-packages', '.venv', 'venv'}

        for obj in objects:
            cls = obj if isinstance(obj, type) else type(obj)
            module_name = cast(str, getattr(cls, '__module__', None))

            if not module_name or module_name in ignored_names or module_name.startswith('mhagenta.'):
                continue

            module = inspect.getmodule(cls)
            module_file_str = inspect.getsourcefile(cls) or getattr(module, '__file__', None)
            if not module_file_str:
                continue

            module_file = Path(module_file_str).resolve()
            parts = {part.lower() for part in module_file.parts}
            if parts & venv_parts or any(p.startswith('.venv') or (Path(p) / 'pyvenv.cfg').exists() for p in parts):
                continue
            if any(module_file.is_relative_to(stdlib_dir) for stdlib_dir in stdlib_dirs):
                continue

            # root = self._module_source_root(module_name, module_file)
            # top_name = module_name.split('.', 1)[0]
            #
            # src = root / top_name
            # if not src.exists():
            #     src = root / f'{top_name}.py'
            # if not src.exists():
            #     raise FileNotFoundError(f'Could not resolve runtime source for module "{module_name}" from "{module_file}": "{src}')
            #
            # self._copy_runtime_sources(src, dst_dir, copied)

            root = self._module_source_root(module_name, module_file)
            module_parts = tuple(module_name.split("."))

            if module_file.name == "__init__.py":
                package_parts = module_parts
            else:
                package_parts = module_parts[:-1]

            if package_parts:
                self._copy_runtime_package_subtree(
                    root=root,
                    package_parts=package_parts,
                    dst_dir=dst_dir,
                    copied=copied,
                )
            else:
                src = root / f"{module_parts[0]}.py"
                if not src.exists():
                    raise FileNotFoundError(
                        f'Could not resolve runtime source for module "{module_name}" from "{module_file}": "{src}"'
                    )
                self._copy_runtime_sources(src, dst_dir, copied)
