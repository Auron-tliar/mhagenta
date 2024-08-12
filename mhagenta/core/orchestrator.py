import asyncio
import inspect
import logging
import types
from typing import Any, Type, Iterable, Callable
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
from mhagenta.core.connection import Connector
from mhagenta.base import *
from mhagenta.containers import *


@dataclass
class AgentEntry:
    agent_id: str
    kwargs: dict[str, Any]
    dir: Path | None = None
    image: Image | None = None
    container: Container | None = None


class Orchestrator:
    SAVE_SUBDIR = 'out/save'
    LOG_CHECK_FREQ = 1.

    def __init__(self,
                 save_dir: str | Path,
                 log_level: int = logging.DEBUG,
                 simulation_duration_sec: float = 60.,
                 step_frequency: float = 1.,
                 control_frequency: float = -1.,
                 status_period: int = 10,
                 start_time: float | None = None,
                 start_sync_delay: float = 2.,
                 ):
        self._agents: dict[str, AgentEntry] = dict()

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir = save_dir

        self._package_dir = str(Path(mhagenta.__file__).parent.absolute())

        self._log_level = log_level
        self._simulation_duration_sec = simulation_duration_sec
        self._step_frequency = step_frequency
        self._control_frequency = control_frequency
        self._status_period = status_period
        self._start_time = start_time
        self._start_sync_delay = start_sync_delay

        self._simulation_end_ts = -1.

        self._docker_client: docker.DockerClient | None = None
        self._rabbitmq_image: Image | None = None
        self._base_image: Image | None = None

        self._docker_init()

        self._stopping = False
        self._all_stopped = False
        self._lock = Lock()

    def _docker_init(self):
        self._docker_client = docker.from_env()

    def add_agent(self,
                  agent_id: str,
                  connector_cls: type[Connector],
                  perceptors: Iterable[PerceptorBase] | PerceptorBase,
                  actuators: Iterable[ActuatorBase] | ActuatorBase,
                  ll_reasoners: Iterable[LLReasonerBase] | LLReasonerBase,
                  learners: Iterable[LearnerBase] | LearnerBase | None = None,
                  knowledge: Iterable[KnowledgeBase] | KnowledgeBase | None = None,
                  hl_reasoner: Iterable[HLReasonerBase] | HLReasonerBase | None = None,
                  goal_graphs: Iterable[GoalGraphBase] | GoalGraphBase | None = None,
                  memory: Iterable[MemoryBase] | MemoryBase | None = None,
                  simulation_duration_sec: float | None = None,
                  step_frequency: float | None = None,
                  control_frequency: float | None = None,
                  status_period: int | None = None,
                  start_time: float | None = None,
                  start_sync_delay: float | None = None,
                  log_level: int | None = None
                  ) -> None:
        kwargs = {
            'agent_id': agent_id,
            'module_connector_cls': module_connector_cls,
            'agent_connector_cls': agent_connector_cls,
            'connector_kwargs': connector_kwargs,
            'simulation_duration_sec': self._simulation_duration_sec if simulation_duration_sec is None else simulation_duration_sec,
            'step_frequency': self._step_frequency if step_frequency is None else step_frequency,
            'control_frequency': self._control_frequency if control_frequency is None else control_frequency,
            'status_period': self._status_period if status_period is None else status_period,
            'start_time': self._start_time if start_time is None else start_time,
            'start_sync_delay': self._start_sync_delay if start_sync_delay is None else start_sync_delay,
            'log_level': self._log_level if log_level is None else log_level,
            'save_dir': f'/{self.SAVE_SUBDIR}'
        }

        if isinstance(perceptors, PerceptorParams):
            perceptors = [perceptors]
        else:
            perceptors = list(perceptors)
        kwargs['perceptors'] = [self._params_to_dict(perceptor) for perceptor in perceptors]

        if isinstance(actuators, ActuatorParams):
            actuators = [actuators]
        else:
            actuators = list(actuators)
        kwargs['actuators'] = [self._params_to_dict(actuator) for actuator in actuators]

        kwargs['ll_reasoner'] = self._params_to_dict(ll_reasoner)
        kwargs['learner'] = self._params_to_dict(learner)
        kwargs['memory'] = self._params_to_dict(memory)
        kwargs['knowledge'] = self._params_to_dict(knowledge)
        kwargs['hl_reasoner'] = self._params_to_dict(hl_reasoner)
        kwargs['goal_graph'] = self._params_to_dict(goal_graph)

        self._agents[agent_id] = AgentEntry(
            agent_id=agent_id,
            kwargs = kwargs
        )

    @staticmethod
    def _params_to_dict(params: ModuleParams | None) -> dict[str, Any] | None:
        if params is None:
            return None
        params = params.model_dump()
        for attr_name, attr_value in params.items():
            if isinstance(attr_value, types.LambdaType):  # and attr_value.__name__ != '<lambda>':
                if attr_value.__name__ != '<lambda>':
                    params[attr_name] = inspect.getsource(attr_value)
                else:
                    source = inspect.getsource(attr_value).strip()
                    if source[-1] == ',':
                        source = source[:-1]
                    compiled = compile(source, '<string>', 'exec')
                    import __main__
                    exec(compiled, __main__.__dict__)
                    params[attr_name] = getattr(__main__, attr_name)
        return params

    def add_agents(self,
                   num_agents: int,
                   agent_id_base: str,
                   ):
        raise NotImplementedError

    def docker_build_base(self,
                          rabbitmq_image_name: str = 'mha-rabbitmq',
                          hagent_base_image_name: str = 'mha-base',
                          version_tag: str = 'test'
                          ):
        print(f'===== BUILDING RABBITMQ BASE IMAGE: {rabbitmq_image_name}:{version_tag} =====')
        if self._rabbitmq_image is None:
            self._rabbitmq_image, _ = self._docker_client.images.build(path=RABBIT_IMG_PATH,
                                                                    tag=f'{rabbitmq_image_name}:{version_tag}',
                                                                    rm=True,
                                                                    quiet=False
                                                                    )  # squash=True

        print(f'===== BUILDING AGENT BASE IMAGE: {hagent_base_image_name}:{version_tag} =====')
        if self._base_image is None:
            build_dir = self._save_dir.absolute() / 'tmp/'
            shutil.copytree(BASE_IMG_PATH, build_dir)
            shutil.copytree(self._package_dir, build_dir / 'mhagenta')

            self._base_image, _ = self._docker_client.images.build(path=str(build_dir),
                                                                buildargs={
                                                                    'SRC_IMAGE': rabbitmq_image_name,
                                                                    'SRC_VERSION': version_tag
                                                                },
                                                                tag=f'{hagent_base_image_name}:{version_tag}',
                                                                rm=True,
                                                                quiet=False
                                                                )  # squash=True
            shutil.rmtree(build_dir)

    def _docker_build_agent(self, agent: AgentEntry, start_delay: float = 5.):
        print(f'===== BUILDING AGENT IMAGE: {agent.agent_id}:test =====')
        agent_dir = self._save_dir.absolute() / agent.agent_id
        (agent_dir / 'out/').mkdir(parents=True)
        agent.dir = agent_dir

        build_dir = agent_dir / 'tmp/'
        shutil.copytree(AGENT_IMG_PATH, build_dir.absolute())
        shutil.copy(Path(mhagenta.core.__file__).parent.absolute() / 'agent_launcher.py', (build_dir / 'src/').absolute())
        shutil.copy(Path(mhagenta.__file__).parent.absolute() / 'scripts/start.sh', (build_dir / 'src/').absolute())

        if agent.kwargs['start_time'] is None:
            agent.kwargs['start_time'] = time.time()
        agent.kwargs['start_time'] += start_delay

        end_estimate = agent.kwargs['start_time'] + agent.kwargs['start_sync_delay'] + agent.kwargs['simulation_duration_sec']
        if self._simulation_end_ts < end_estimate:
            self._simulation_end_ts = end_estimate

        with open((build_dir / 'src/agent_params').absolute(), 'wb') as f:
            dill.dump(agent.kwargs, f)

        base_tag = self._base_image.tags[0].split(':')
        agent.image, _ = self._docker_client.images.build(path=str(build_dir.absolute()),
                                                       buildargs={
                                                           'SRC_IMAGE': base_tag[0],
                                                           'SRC_VERSION': base_tag[1]
                                                       },
                                                       tag=f'{agent.agent_id}:test',
                                                       rm=True,
                                                       quiet=False
                                                       )  # squash=True
        shutil.rmtree(build_dir)

    async def _run_agent(self,
                         agent: AgentEntry,
                         force_run: bool = False):
        print(f'===== RUNNING AGENT IMAGE AS CONTAINER \"{agent.agent_id}\" =====')
        try:
            container = self._docker_client.containers.get(agent.agent_id)
            if force_run:
                container.remove(force=True)
            else:
                raise NameError(f'Container {agent.agent_id} already exists')
        except NotFound:
            pass

        agent.container = self._docker_client.containers.start(image=agent.image,
                                                               detach=True,
                                                               name=agent.agent_id,
                                                               volumes={
                                                                 str((agent.dir / "out").absolute()): {'bind': '/out', 'mode': 'rw'}
                                                             })

    async def run(self,
                  start_delay: float = 5.,
                  rabbitmq_image_name: str = 'mha-rabbitmq',
                  hagent_base_image_name: str = 'mha-base',
                  force_run: bool = False
                  ):
        assert start_delay >= 0

        if self._base_image is None:
            self.docker_build_base(rabbitmq_image_name=rabbitmq_image_name, hagent_base_image_name=hagent_base_image_name)

        for agent in self._agents.values():
            self._docker_build_agent(agent, start_delay=start_delay)

        async with asyncio.TaskGroup() as tg:
            for agent in self._agents.values():
                tg.create_task(self._run_agent(agent, force_run=force_run))
                tg.create_task(self.simulation_end_timer())
                tg.create_task(self._read_logs(agent))
        for agent in self._agents.values():
            agent.container.remove()
        print('===== EXECUTION FINISHED =====')

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
        with self._lock:
            self._stopping = True

    def add_log(self, log: str | bytes):
        if isinstance(log, bytes):
            log = log.decode()
        print(log.strip('\n\r'))

    async def _read_logs(self, agent: AgentEntry):
        logs = self._docker_client.containers.get(agent.container.id).logs(stdout=True, stderr=True, stream=True, follow=True)

        while True:
            with self._lock:
                if self._stopping and self._agents_stopped:
                    break
            for line in logs:
                self.add_log(line)
            await asyncio.sleep(self.LOG_CHECK_FREQ)
