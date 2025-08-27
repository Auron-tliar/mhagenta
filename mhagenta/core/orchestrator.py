import asyncio
import logging
import os
import shutil
import sys
import time
from asyncio import TaskGroup
from dataclasses import dataclass
from io import TextIOWrapper
from os import PathLike
from datetime import datetime, timedelta
import socket
from pathlib import Path
from typing import Any, Iterable, Literal, Self, Callable
import subprocess
import signal
import functools
import dateutil.parser
import dateutil.tz
from pprint import pprint

import dill
import docker
import pika
# from Demos.c_extension.setup import sources
from pika.adapters import BlockingConnection
from pika.channel import Channel
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
# from mhagenta.utils.common.classes import EDirectory
from mhagenta.gui import Monitor
from mhagenta.utils.common.classes import EDirectory


@dataclass
class AgentEntry:
    agent_id: str
    kwargs: dict[str, Any]
    dir: Path | None = None
    save_dir: Path | None = None
    image: Image | None = None
    container: Container | None = None
    port_mapping: dict[int, int] | None = None
    num_copies: int = 1
    save_logs: bool = True
    tags: Iterable[str] | None = None

    @property
    def module_ids(self) -> list[str]:
        module_ids = []
        keys = ('perceptors',
                'actuators',
                'll_reasoners',
                'learners',
                'knowledge',
                'hl_reasoners',
                'goal_graphs',
                'memory')
        for key in keys:
            if self.kwargs[key] is None:
                continue
            if isinstance(self.kwargs[key], Iterable):
                for module in self.kwargs[key]:
                    module_ids.append(module.module_id)
            else:
                module_ids.append(self.kwargs[key].module_id)
        return module_ids

    def __hash__(self) -> int:
        return hash(self.agent_id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AgentEntry):
            return False
        return self.agent_id == other.agent_id

@dataclass
class EnvironmentEntry:
    env_id: str
    kwargs: dict[str, Any]
    address: dict[str, Any]
    dir: Path | None = None
    tags: list[str] | None = None
    # process: subprocess.Popen | None = None
    image: Image | None = None
    container: Container | None = None
    port_mapping: dict[int, int] | None = None

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

    def __init__(self, stop_checker: Callable[[], bool], check_freq: float = 1., save_logs: PathLike | str | None = None) -> None:
        # self._containers: list[AgentEntry | EnvironmentEntry] = list()
        self._sources: dict[AgentEntry | EnvironmentEntry, LogParser.SourceInfo] = dict()
        self._check_freq: float = check_freq
        self._stop_checker: Callable[[], bool] = stop_checker
        self._save_logs: Path | None = Path(save_logs) if save_logs else None

        self._init_ts = datetime.now()

    def add_container(self, source: AgentEntry | EnvironmentEntry) -> None:
        sid: str
        if isinstance(source, EnvironmentEntry):
            sid = source.env_id
        else:
            sid = source.agent_id
        self._sources[source] = self.SourceInfo(
            last_ts=self._init_ts,
            path=(self._save_logs / f'{sid}.log') if self._save_logs is not None else None
        )

    @staticmethod
    def _add_log(log: str | bytes, save_path: Path | None = None) -> None:
        if isinstance(log, bytes):
            log = log.decode().strip('\n\r')
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
            for source, info in self._sources.items():
                raw_log = source.container.logs(stdout=True, stderr=True, tail='all', timestamps=True, since=(info.last_ts + self.US).timestamp())
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
                self._add_log(entry.msg, save_path=entry.source.path)
            logs.clear()
            await asyncio.sleep(self._check_freq)


class Orchestrator:
    """Orchestrator class that handles MHAgentA execution.

    Orchestrator handles definition of agents and their consequent containerization and deployment. It also allows you
    to define default parameters shared by all the agents handles by it (can be overridden by individual agents)

    """
    SAVE_SUBDIR = 'out/save'
    LOG_CHECK_FREQ = 1.

    def __init__(self,
                 save_dir: str | PathLike,
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
                 module_start_delay: float = 2.,
                 connector_cls: type[Connector] = RabbitMQConnector,
                 connector_kwargs: dict[str, Any] | None = None,
                 mas_rmq_uri: str | Literal['default'] | None = None,
                 mas_rmq_close_on_exit: bool = True,
                 mas_rmq_exchange_name: str | None = None,
                 save_logs: bool = True
                 ) -> None:
        """
        Constructor method for Orchestrator.

        Args:
            save_dir (str | PathLike): Root directory for storing agents' states, logs, and temporary files.
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
        self._module_start_delay = module_start_delay
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
        if 'localhost' in self._mas_rmq_uri_internal:
            self._mas_rmq_uri_internal = self._mas_rmq_uri_internal.replace('localhost', EDirectory.localhost_linux if sys.platform == 'linux' else EDirectory.localhost_win)
        self._mas_rmq_close_on_exit = mas_rmq_close_on_exit
        self._mas_rmq_container: Container | None = None
        self._mas_rmq_exchange_name = mas_rmq_exchange_name

        self._start_time: float = -1.
        self._simulation_end_ts = -1.

        self._docker_client: docker.DockerClient | None = None
        self._rabbitmq_image: Image | None = None
        self._base_image: Image | None = None

        self._task_group: TaskGroup | None = None
        self._force_run = False

        self._docker_init()

        self._monitor: Monitor | None = None

        self._running = False
        self._stopping = False
        self._all_stopped = False

        self._log_parser = LogParser(
            stop_checker=lambda: self._stopping and self._agents_stopped,
            check_freq=1.,
            save_logs=self._save_dir
        )

    def _docker_init(self) -> None:
        self._docker_client = docker.from_env()

    def add_environment(self,
                        base: MHAEnvBase,
                        env_id: str = "environment",
                        host: str | None = 'localhost',
                        port: int | None = 5672,
                        exec_duration: float | None = None,
                        exchange_name: str | None = None,
                        init_script: PathLike | str | None = None,
                        requirements_path: PathLike | str | None = None,
                        port_mapping: dict[int, int] | None = None,
                        log_tags: list[str] | None = None,
                        log_level: int | str | None = None,
                        log_format: str | None = None,
                        tags: Iterable[str] | None = None
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
            init_script (PathLike | str, optional): Path to an optional bash script to be run before launching the environment.
                Use it to install additional non-Python dependencies. Defaults to None.
            requirements_path (PathLike | str, optional): Path to Python dependencies file. Defaults to None.
            port_mapping (dict[int, int], optional): Mapping between internal docker container ports and host ports.
                Defaults to the Orchestrator's `port_mapping`.
            log_tags (list[str], optional): List of tags to add to log messages. Defaults to None.
            log_level (int, optional): Log level; will use the Orchestrator's log level in None. Defaults to None.
            log_format (str, optional): Log format string; will use the Orchestrator's log format. Defaults to None.
            tags (Iterable[str], optional): List of tags for agent directory. Defaults to None.

        Returns:

        """
        from mhagenta.defaults.communication.rabbitmq import RMQEnvironment
        if host is None:
            mas_host, mas_port = self._mas_rmq_uri.split(':')
            host = mas_host
            if port is None:
                port = mas_port
        env_dir = self._save_dir.resolve() / env_id
        env_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            'env_class': RMQEnvironment,
            'base': base,
            'env_id': env_id,
            'host': host if host != 'localhost' and host != '127.0.0.1' else 'host.docker.internal',
            'port': port,
            'exec_duration': (exec_duration if exec_duration else self._exec_duration_sec) + self._agent_start_delay,
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

        if requirements_path is not None:
            kwargs['requirements_path'] = Path(requirements_path).resolve()
        if init_script is not None:
            kwargs['init_script'] = Path(init_script).resolve()

        self._environments[env_id] = EnvironmentEntry(
            env_id=env_id,
            kwargs=kwargs,
            address={
                'exchange_name': kwargs['exchange_name'],
                'env_id': env_id
            },
            dir=env_dir,
            tags=tags,
            port_mapping=port_mapping if port_mapping else self._port_mapping
        )

    def _update_external_host(self, module: ActuatorBase | PerceptorBase):
        if 'external' in module.tags and module.conn_params['host'] == 'localhost':
            module.conn_params['host'] = EDirectory.localhost_linux if sys.platform == 'linux' else EDirectory.localhost_win

    def add_agent(self,
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
                  init_script: PathLike | str | None = None,
                  requirements_path: PathLike | str | None = None,
                  log_level: int | None = None,
                  port_mapping: dict[int, int] | None = None,
                  connector_cls: type[Connector] | None = None,
                  connector_kwargs: dict[str, Any] | None = None,
                  save_logs: bool | None = None,
                  tags: Iterable[str] | None = None
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
            init_script (PathLike | str, optional): Path to an optional bash script to be run before launching the agent.
                Use it to install additional non-Python dependencies. Defaults to None.
            requirements_path (PathLike | str, optional): Additional Python requirements to install on agent side.
            log_level (int, optional):  Logging level for the agent. Defaults to the Orchestrator's `log_level`.
            port_mapping (dict[int, int], optional): Mapping between internal docker container ports and host ports.
                Defaults to the Orchestrator's `port_mapping`.
            connector_cls (type[Connector], optional): internal connector class that implements communication between
                modules. Defaults to the Orchestrator's `connector_cls`.
            connector_kwargs (dict[str, Any], optional): Additional keyword arguments for connector. Defaults to
                the Orchestrator's `connector_kwargs`.
            save_logs (bool, optional): Whether to save agent logs. If True, saves the agent's logs to
                `<agent_id>.log` at the root of the `save_dir`. Defaults to the orchestrator's `save_logs`.
            tags (Iterable[str], optional): a list of tags associated with this agent for directory search.

        """
        if isinstance(actuators, Iterable):
            for actuator in actuators:
                self._update_external_host(actuator)
        else:
            self._update_external_host(actuators)

        if isinstance(perceptors, Iterable):
            for perceptor in perceptors:
                self._update_external_host(perceptor)
        else:
            self._update_external_host(perceptors)

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

        if init_script is not None:
            kwargs['init_script'] = Path(init_script).resolve()
        if requirements_path is not None:
            kwargs['requirements_path'] = Path(requirements_path).resolve()

        self._agents[agent_id] = AgentEntry(
            agent_id=agent_id,
            port_mapping=port_mapping if port_mapping else self._port_mapping,
            num_copies=num_copies,
            kwargs=kwargs,
            save_logs=save_logs if save_logs is not None else self._save_logs,
            tags=tags
        )
        if self._task_group is not None:
            self._task_group.create_task(self._run_agent(self._agents[agent_id], force_run=self._force_run))

    def _compose_directory(self) -> Directory:
        directory = Directory()
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
            directory.external.add_agent(
                agent_id=agent.agent_id,
                address={
                    'exchange_name': self._mas_rmq_exchange_name,
                    'agent_id': agent.agent_id,
                    'host': host,
                    'port': port
                },
                tags=agent.tags
            )
        return directory

    def _docker_build_base(self,
                           mhagenta_version: str = 'latest',
                           local_build: PathLike | str | None = None,
                           prerelease: bool = False
                           ) -> None:
        if not mhagenta_version:
            mhagenta_version = CONTAINER_VERSION
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
                                'PRE_VERSION': "true" if prerelease else "false",
                                'LOCAL': "false" if local_build is None else "true",
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

    def _docker_build_agent(self,
                            agent: AgentEntry,
                            rebuild_image: bool = True,
                            ) -> None:
        if rebuild_image:
            try:
                img = self._docker_client.images.list(name=f'mhagent:{agent.agent_id}')[0]
                img.remove(force=True)
            except IndexError:
                pass
        print(f'===== BUILDING AGENT IMAGE: mhagent:{agent.agent_id} =====')
        agent_dir = self._save_dir.resolve() / agent.agent_id
        if self._force_run and agent_dir.exists():
            shutil.rmtree(agent_dir)

        (agent_dir / 'out/').mkdir(parents=True)
        agent.dir = agent_dir
        agent.save_dir = agent_dir / 'out' / 'save'

        build_dir = agent_dir / 'tmp/'
        shutil.copytree(AGENT_IMG_PATH, build_dir.resolve())
        (build_dir / 'src').mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(mhagenta.core.__file__).parent.resolve() / 'agent_launcher.py', (build_dir / 'src' / 'agent_launcher.py').resolve())
        shutil.copy(Path(mhagenta.__file__).parent.resolve() / 'scripts' / 'start.sh', (build_dir / 'src' / 'start.sh').resolve())

        agent.kwargs['directory'] = self._compose_directory()

        if agent.kwargs['exec_start_time'] is None:
            agent.kwargs['exec_start_time'] = self._start_time

        agent.kwargs['exec_start_time'] += self._agent_start_delay

        end_estimate = agent.kwargs['exec_start_time'] + agent.kwargs['start_delay'] + agent.kwargs['exec_duration']
        if self._simulation_end_ts < end_estimate:
            self._simulation_end_ts = end_estimate

        if 'init_script' in agent.kwargs:
            init_script = agent.kwargs.pop('init_script')
            shutil.copy(init_script, (build_dir / 'src' / 'init_script.sh').resolve())

        if 'requirements_path' in agent.kwargs:
            requirements_path = agent.kwargs.pop('requirements_path')
            shutil.copy(requirements_path, (build_dir / 'src' / 'requirements.txt').resolve())

        with open((build_dir / 'src' / 'agent_params').resolve(), 'wb') as f:
            dill.dump(agent.kwargs, f, recurse=True)

        base_tag = self._base_image.tags[0].split(':')
        agent.image, _ = self._logged_build(path=str(build_dir.resolve()),
                                            buildargs={
                                                'SRC_IMAGE': base_tag[0],
                                                'SRC_VERSION': base_tag[1]
                                            },
                                            tag=f'mhagent:{agent.agent_id}',
                                            rm=True,
                                            quiet=False
                                            )
        shutil.rmtree(build_dir)

    def _docker_build_env(self,
                          environment: EnvironmentEntry,
                          rebuild_image: bool = True,
                            ) -> None:
        if rebuild_image:
            try:
                img = self._docker_client.images.list(name=f'mhagent-env:{environment.env_id}')[0]
                img.remove(force=True)
            except IndexError:
                pass
        print(f'===== BUILDING ENVIRONMENT IMAGE: mhagent-env:{environment.env_id} =====')
        env_dir = self._save_dir.resolve() / environment.env_id
        if self._force_run and env_dir.exists():
            shutil.rmtree(env_dir)

        (env_dir / 'out/').mkdir(parents=True)
        environment.dir = env_dir

        build_dir = env_dir / 'tmp/'
        shutil.copytree(AGENT_IMG_PATH, build_dir.resolve())
        (build_dir / 'src').mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(mhagenta.environment.__file__).parent.resolve() / 'environment_launcher.py', (build_dir / 'src' / 'environment_launcher.py').resolve())
        shutil.copy(Path(mhagenta.__file__).parent.resolve() / 'scripts' / 'env_start.sh', (build_dir / 'src' / 'start.sh').resolve())

        if 'init_script' in environment.kwargs:
            init_script = environment.kwargs.pop('init_script')
            shutil.copy(init_script, (build_dir / 'src' / 'init_script.sh').resolve())

        if 'requirements_path' in environment.kwargs:
            requirements_path = environment.kwargs.pop('requirements_path')
            shutil.copy(requirements_path, (build_dir / 'src' / 'requirements.txt').resolve())

        with open((build_dir / 'src' / 'env_params').resolve(), 'wb') as f:
            dill.dump(environment.kwargs, f, recurse=True)

        base_tag = self._base_image.tags[0].split(':')
        environment.image, build_logs = self._logged_build(path=str(build_dir.resolve()),
                                                           buildargs={
                                                               'SRC_IMAGE': base_tag[0],
                                                               'SRC_VERSION': base_tag[1]
                                                           },
                                                           tag=f'mhagent-env:{environment.env_id}',
                                                           rm=True,
                                                           quiet=False,
                                                           nocache=True
                                                           )
        shutil.rmtree(build_dir)

    async def _run_agent(self,
                         agent: AgentEntry,
                         force_run: bool = False
                         ) -> None:
        if agent.num_copies == 1:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS CONTAINER \"{agent.agent_id}\" =====')
        else:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS '
                  f'{agent.num_copies} CONTAINERS \"{agent.agent_id}_#\" =====')
        for i in range(agent.num_copies):
            if agent.num_copies == 1:
                agent_name = agent.agent_id
                agent_dir = (agent.dir / "out").resolve()
            else:
                agent_name = f'{agent.agent_id}_{i}'
                agent_dir = (agent.dir / str(i) / "out").resolve()

            agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                container = self._docker_client.containers.get(agent_name)
                if force_run:
                    container.remove(force=True)
                else:
                    raise NameError(f'Container {agent_name} already exists')
            except NotFound:
                pass

            agent.container = self._docker_client.containers.run(
                image=agent.image,
                detach=True,
                name=agent_name,
                environment={"AGENT_ID": agent_name},
                volumes={
                    str(agent_dir): {'bind': '/out', 'mode': 'rw'}
                },
                extra_hosts={'host.docker.internal': 'host-gateway'},
                ports=agent.port_mapping
            )

    async def _run_env(self,
                         environment: EnvironmentEntry,
                         force_run: bool = False
                         ) -> None:
        print(f'===== RUNNING ENVIRONMENT IMAGE \"mhagent-env:{environment.env_id}\" AS CONTAINER \"{environment.env_id}\" =====')

        env_dir = (environment.dir / "out").resolve()
        env_dir.mkdir(parents=True, exist_ok=True)

        try:
            container = self._docker_client.containers.get(environment.env_id)
            if force_run:
                container.remove(force=True)
            else:
                raise NameError(f'Container {environment.env_id} already exists')
        except NotFound:
            pass

        environment.container = self._docker_client.containers.run(
            image=environment.image,
            detach=True,
            name=environment.env_id,
            environment={"AGENT_ID": ''},
            volumes={
                str(env_dir): {'bind': '/out', 'mode': 'rw'}
            },
            extra_hosts={'host.docker.internal': 'host-gateway'},
            ports=environment.port_mapping
        )

    async def arun(self,
                   mhagenta_version: str = 'latest',
                   force_run: bool = False,
                   gui: bool = False,
                   rebuild_agents: bool = True,
                   local_build: PathLike | str | None = None,
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
            rebuild_agents (bool, optional, default=True): Whether to rebuild the agents. Defaults to True.
            local_build (PathLike | str, optional): Specifies the path to a local build of MHAgentA (as opposed to the latest
                one from PyPI) to be used for building agents.
            prerelease (bool, optional, default=False): Specifies whether to allow agents to use the latest prerelease
                version of mhagenta while building the container.
            keep_containers (bool, optional, default=False): Whether to keep or remove the agent and environment
                containers after the execution.

        Raises:
            NameError: Raised if a container for one of the specified agent IDs already exists and `force_run` is False.

        """

        self._start_time = time.time()
        if self._base_image is None:
            self._docker_build_base(mhagenta_version=mhagenta_version, local_build=local_build, prerelease=prerelease)

        self._force_run = force_run
        for env in self._environments.values():
            self._docker_build_env(env, rebuild_image=rebuild_agents)
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
                env.kwargs['exec_duration'] -= (time.time() - self._start_time)
                tg.create_task(self._run_env(env, force_run=force_run))
                # tg.create_task(self._read_logs(env, gui=False))
                self._log_parser.add_container(env)
            for agent in self._agents.values():
                tg.create_task(self._run_agent(agent, force_run=force_run))
                # tg.create_task(self._read_logs(agent, gui))
                self._log_parser.add_container(agent)
            tg.create_task(self._log_parser.run())
        self._running = False
        for agent in self._agents.values():
            agent.container.stop()
            if not keep_containers:
                agent.container.remove()
        for env in self._environments.values():
            env.container.stop()
            if not keep_containers:
                env.container.remove()
        if self._mas_rmq_container is not None and self._mas_rmq_close_on_exit:
            try:
                self._mas_rmq_container.stop()
            except Exception:
                pass
        print('===== EXECUTION FINISHED =====')

    def run(self,
            mhagenta_version='latest',
            force_run: bool = False,
            gui: bool = False,
            rebuild_agents: bool = True,
            local_build: PathLike | str | None = None,
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
            local_build (PathLike | str, optional): Specifies the path to a local build of MHAgentA (as opposed to the latest
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
            local_build=local_build,
            prerelease=prerelease,
            keep_containers=keep_containers
        ))

    @staticmethod
    def _agent_stopped(agent: AgentEntry | EnvironmentEntry) -> bool:
        agent.container.reload()
        return agent.container.status == 'exited'

    @property
    def _agents_stopped(self) -> bool:
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
                    '15672':15672
                },
                remove=True,
                tty=True
            )
