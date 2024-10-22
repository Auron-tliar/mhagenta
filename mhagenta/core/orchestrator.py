import asyncio
import inspect
import logging
import os
import types
from asyncio import TaskGroup
from typing import Any, Type, Iterable, Callable, Literal
from pathlib import Path
import time
from dataclasses import dataclass
from pydantic import BaseModel
import dill
import docker
from docker.errors import NotFound, ContextAlreadyExists
import shutil
from threading import Lock
from pprint import pprint
from docker.models.images import Image
from docker.models.containers import Container

import mhagenta
from mhagenta.core.connection import Connector, RabbitMQConnector
from mhagenta.base import *
from mhagenta.containers import *
from mhagenta.utils.common import DEFAULT_LOG_FORMAT


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


class Orchestrator:
    SAVE_SUBDIR = 'out/save'
    LOG_CHECK_FREQ = 1.

    def __init__(self,
                 save_dir: str | Path,
                 connector_cls: type[Connector] = RabbitMQConnector,
                 connector_kwargs: dict[str, Any] | None = None,
                 port_mapping: dict[int, int] | None = None,
                 step_frequency: float = 1.,
                 status_frequency: float = 10,
                 control_frequency: float = -1.,
                 module_start_delay: float = 2.,
                 exec_start_time: float | None = None,
                 exec_duration: float = 60.,
                 agent_start_delay: float = 5.,
                 save_format: Literal['json', 'dill'] = 'json',
                 resume: bool = False,
                 log_level: int = logging.INFO,
                 log_format: str | None = None,
                 status_msg_format: str = '[status_upd]::{}'
                 ):
        self._agents: dict[str, AgentEntry] = dict()

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir = save_dir

        self._package_dir = str(Path(mhagenta.__file__).parent.absolute())

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

        self._simulation_end_ts = -1.

        self._docker_client: docker.DockerClient | None = None
        self._rabbitmq_image: Image | None = None
        self._base_image: Image | None = None

        self._task_group: TaskGroup | None = None
        self._force_run = False

        self._docker_init()

        self._running = False
        self._stopping = False
        self._all_stopped = False

    def _docker_init(self):
        self._docker_client = docker.from_env()

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
                  connector_cls: type[Connector] | None = None,
                  connector_kwargs: dict[str, Any] | None = None,
                  step_frequency: float | None = None,
                  status_frequency: float | None = None,
                  control_frequency: float | None = None,
                  exec_start_time: float | None = None,
                  start_delay: float | None = None,
                  exec_duration: float | None = None,
                  resume: bool | None = None,
                  log_level: int | None = None,
                  port_mapping: dict[int, int] | None = None
                  ) -> None:
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
            'start_delay': self._module_start_delay if start_delay is None else start_delay,
            'exec_duration': self._exec_duration_sec if exec_duration is None else exec_duration,
            'save_dir': f'/{self.SAVE_SUBDIR}/{agent_id}',
            'save_format': self._save_format,
            'resume': self._resume if resume is None else resume,
            'log_level': self._log_level if log_level is None else log_level,
            'log_format': self._log_format,
            'status_msg_format': self._status_msg_format
        }

        # ports = port_mapping if port_mapping else self._port_mapping
        self._agents[agent_id] = AgentEntry(
            agent_id=agent_id,
            port_mapping=port_mapping if port_mapping else self._port_mapping,
            num_copies=num_copies,
            kwargs=kwargs
        )
        if self._task_group is not None:
            self._task_group.create_task(self._run_agent(self._agents[agent_id], force_run=self._force_run))

    def add_agents(self,
                   num_agents: int,
                   agent_id_base: str,
                   ):
        raise NotImplementedError()

    def docker_build_base(self,
                          rabbitmq_image_name: str = 'mha-rabbitmq',
                          mha_base_image_name: str = 'mha-base',
                          version_tag: str | None = None
                          ):
        if version_tag is None:
            version_tag = CONTAINER_VERSION
        print(f'===== BUILDING RABBITMQ BASE IMAGE: {rabbitmq_image_name}:{version_tag} =====')
        if self._rabbitmq_image is None:
            self._rabbitmq_image, _ = (
                self._docker_client.images.build(path=RABBIT_IMG_PATH,
                                                 tag=f'{rabbitmq_image_name}:{version_tag}',
                                                 rm=True,
                                                 quiet=False
                                                 ))

        print(f'===== BUILDING AGENT BASE IMAGE: {mha_base_image_name}:{version_tag} =====')
        if self._base_image is None:
            try:
                self._base_image = self._docker_client.images.list(name=f'{mha_base_image_name}:{version_tag}')[0]
            except IndexError:
                build_dir = self._save_dir.absolute() / 'tmp/'
                try:
                    shutil.copytree(BASE_IMG_PATH, build_dir)
                    shutil.copytree(self._package_dir, build_dir / 'mhagenta')

                    self._base_image, _ = (
                        self._docker_client.images.build(path=str(build_dir),
                                                         buildargs={
                                                             'SRC_IMAGE': rabbitmq_image_name,
                                                             'SRC_VERSION': version_tag
                                                         },
                                                         tag=f'{mha_base_image_name}:{version_tag}',
                                                         rm=True,
                                                         quiet=False
                                                         ))
                except Exception as ex:
                    shutil.rmtree(build_dir)
                    raise ex
                shutil.rmtree(build_dir)

    def _docker_build_agent(self,
                            agent: AgentEntry
                            ):
        print(f'===== BUILDING AGENT IMAGE: mhagent:{agent.agent_id} =====')
        agent_dir = self._save_dir.absolute() / agent.agent_id
        if self._force_run and agent_dir.exists():
            shutil.rmtree(agent_dir)

        (agent_dir / 'out/').mkdir(parents=True)
        agent.dir = agent_dir
        agent.save_dir = agent_dir / 'out' / 'save' / agent.agent_id

        build_dir = agent_dir / 'tmp/'
        shutil.copytree(AGENT_IMG_PATH, build_dir.absolute())
        shutil.copy(Path(mhagenta.core.__file__).parent.absolute() / 'agent_launcher.py', (build_dir / 'src/').absolute())
        shutil.copy(Path(mhagenta.__file__).parent.absolute() / 'scripts/start.sh', (build_dir / 'src/').absolute())

        if agent.kwargs['exec_start_time'] is None:
            agent.kwargs['exec_start_time'] = time.time()
        agent.kwargs['exec_start_time'] += self._agent_start_delay

        end_estimate = agent.kwargs['exec_start_time'] + agent.kwargs['start_delay'] + agent.kwargs['exec_duration']
        if self._simulation_end_ts < end_estimate:
            self._simulation_end_ts = end_estimate

        with open((build_dir / 'src/agent_params').absolute(), 'wb') as f:
            dill.dump(agent.kwargs, f, recurse=True)

        base_tag = self._base_image.tags[0].split(':')
        agent.image, _ = self._docker_client.images.build(path=str(build_dir.absolute()),
                                                          buildargs={
                                                              'SRC_IMAGE': base_tag[0],
                                                              'SRC_VERSION': base_tag[1]
                                                          },
                                                          tag=f'mhagent:{agent.agent_id}',
                                                          rm=True,
                                                          quiet=False
                                                          )
        shutil.rmtree(build_dir)

    async def _run_agent(self,
                         agent: AgentEntry,
                         force_run: bool = False):
        if agent.num_copies == 1:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS CONTAINER \"{agent.agent_id}\" =====')
        else:
            print(f'===== RUNNING AGENT IMAGE \"mhagent:{agent.agent_id}\" AS '
                  f'{agent.num_copies} CONTAINERS \"{agent.agent_id}_#\" =====')
        for i in range(agent.num_copies):
            if agent.num_copies == 1:
                agent_name = agent.agent_id
                agent_dir = (agent.dir / "out").absolute()
            else:
                agent_name = f'{agent.agent_id}_{i}'
                agent_dir = (agent.dir / str(i) / "out").absolute()

            agent_dir.mkdir(parents=True)
            try:
                container = self._docker_client.containers.get(agent_name)
                if force_run:
                    container.remove(force=True)
                else:
                    raise NameError(f'Container {agent_name} already exists')
            except NotFound:
                pass

            agent.container = self._docker_client.containers.run(image=agent.image,
                                                                 detach=True,
                                                                 name=agent_name,
                                                                 volumes={
                                                                     str(agent_dir): {'bind': '/out', 'mode': 'rw'}
                                                                 },
                                                                 extra_hosts={'host.docker.internal': 'host-gateway'},
                                                                 ports=agent.port_mapping)

    async def arun(self,
                   rabbitmq_image_name: str = 'mha-rabbitmq',
                   hagent_base_image_name: str = 'mha-base',
                   force_run: bool = False
                   ):
        if self._base_image is None:
            self.docker_build_base(rabbitmq_image_name=rabbitmq_image_name,
                                   mha_base_image_name=hagent_base_image_name)

        self._force_run = force_run
        for agent in self._agents.values():
            self._docker_build_agent(agent)

        self._running = True
        async with asyncio.TaskGroup() as tg:
            self._task_group = tg
            for agent in self._agents.values():
                tg.create_task(self._run_agent(agent, force_run=force_run))
                tg.create_task(self.simulation_end_timer())
                tg.create_task(self._read_logs(agent))
        self._running = False
        for agent in self._agents.values():
            agent.container.remove()
        print('===== EXECUTION FINISHED =====')

    def run(self,
            rabbitmq_image_name: str = 'mha-rabbitmq',
            hagent_base_image_name: str = 'mha-base',
            force_run: bool = False
            ):
        asyncio.run(self.arun(
            rabbitmq_image_name=rabbitmq_image_name,
            hagent_base_image_name=hagent_base_image_name,
            force_run=force_run
        ))

    @staticmethod
    def _agent_stopped(agent: AgentEntry) -> bool:
        agent.container.reload()
        return agent.container.status == 'exited'

    @property
    def _agents_stopped(self) -> bool:
        if self._all_stopped:
            return True
        for agent in self._agents.values():
            if not self._agent_stopped(agent):
                return False
        self._all_stopped = True
        return True

    async def simulation_end_timer(self):
        await asyncio.sleep(self._simulation_end_ts - time.time())
        self._stopping = True

    def add_log(self, log: str | bytes):
        if isinstance(log, bytes):
            log = log.decode()
        print(log.strip('\n\r'))

    async def _read_logs(self, agent: AgentEntry):
        logs = self._docker_client.containers.get(agent.container.id).logs(stdout=True, stderr=True, stream=True, follow=True)

        while True:
            if self._stopping and self._agents_stopped:
                break
            for line in logs:
                self.add_log(line)
            await asyncio.sleep(self.LOG_CHECK_FREQ)

    def __getitem__(self, agent_id: str) -> AgentEntry:
        return self._agents[agent_id]
