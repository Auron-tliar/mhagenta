from pydantic import BaseModel
from typing import Any, Literal, ClassVar, Callable, Self, Iterable
from abc import ABC, abstractmethod
from functools import total_ordering
import heapq
from pathlib import Path
import time
import logging
import sys
from mhagenta.utils.common.logging import ILogging, LoggerExtras
from mhagenta.utils.common.logging import DEFAULT_FORMAT as DEFAULT_LOG_FORMAT
from uuid import uuid4


class MHABase(ILogging, ABC):
    def __init__(self,
                 agent_id: str,
                 log_id: str | None = None,
                 log_tags: list[str] | None = None,
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT
                 ) -> None:
        self._agent_id = agent_id
        self._log_id = self.__class__.__name__ if log_id is None else log_id
        self._log_tags = log_tags[:] if log_tags else [agent_id]
        self._log_tags.append(self._log_id)
        self._log_level = log_level
        self._log_format = log_format

        self._local_logger = logging.getLogger('.'.join(self._log_tags))
        self._local_logger.setLevel(log_level)
        if not self._local_logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(log_format))
            self._local_logger.addHandler(handler)

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @property
    def log_id(self) -> str:
        return self._log_id

    @property
    def log_tag_str(self) -> str:
        return f'[{"][".join(self._log_tags)}]'

    @property
    def _logger(self) -> logging.Logger:
        return self._local_logger


class AgentTime:
    """
    Utility class for getting timestamps for agent components.
    """
    def __init__(self, agent_start_ts: float, exec_start_ts: float | None = None, decimals: int = 4) -> None:
        self._agent_start_ts = agent_start_ts
        self._module_start_ts: float = time.time()
        self._exec_start_ts: float | None = exec_start_ts
        self._decimals = decimals

    @property
    def system(self) -> float:
        """float: System time (in seconds) as provided by `time` module."""
        return round(time.time(), self._decimals)

    @property
    def agent(self) -> float:
        """float: Seconds since the agent was created (i.e. since the initialization of `MHARoot`)."""
        return round(time.time() - self._agent_start_ts, self._decimals)

    @property
    def module(self) -> float:
        """float: Seconds since the initializations of the current module."""
        return round(time.time() - self._module_start_ts, self._decimals)

    @property
    def exec(self) -> float | None:
        """
        float | None: Seconds since the synchronous start of agent execution. Is None if the module doesn't have the
            information yet, or negative if the execution start is scheduled in the future.
         """
        return round(time.time() - self._agent_start_ts - self._exec_start_ts, self._decimals) if self._exec_start_ts is not None else None

    @property
    def agent_start_ts(self) -> float:
        return self._agent_start_ts

    @property
    def exec_start_ts(self) -> float | None:
        return self._exec_start_ts

    def set_exec_start_ts(self, exec_start_ts: float) -> None:
        """
        Pass the information on when the agent execution is scheduled to start.

        Args:
            exec_start_ts: system-level timestamp (in seconds) of agent's scheduled start time.
        """
        self._exec_start_ts = exec_start_ts

    def get_exec_time(self) -> float | None:
        """
        Functional version of `exec` property. Used by execution queues.

        Returns:
            float if defined, None otherwise
        """
        return self.exec

    @staticmethod
    def sleep(secs: float) -> None:
        time.sleep(secs)

    @property
    def tuple(self) -> tuple[float, float, float, float | None]:
        """tuple[float, float, float, float | None]: tuple of all four types of timestamps in order: system, agent,
            module, execution"""
        return self.system, self.agent, self.module, self.exec


class State:
    def __init__(self, agent_id: str, module_id: str, time_func: Callable[[], float], **kwargs) -> None:
        self._agent_id = agent_id
        self._module_id = module_id
        self._time_func = time_func

        self._custom_fields = set(kwargs.keys())
        self.__dict__.update(kwargs)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def module_id(self) -> str:
        return self._module_id

    @property
    def time(self) -> float:
        return self._time_func()

    @property
    def modules(self) -> dict[str, list[str]]:
        raise NotImplementedError

    def dump(self) -> dict[str, Any]:
        return {field: self.__dict__[field] for field in self._custom_fields}

    def load(self, **kwargs) -> None:
        self._custom_fields.update(kwargs.keys())
        self.__dict__.update(kwargs)


class Belief(BaseModel):
    predicate: str
    arguments: tuple[str]
    misc: dict[str, Any]

    def __init__(self, predicate: str, arguments: tuple[str], **kwargs) -> None:
        super().__init__(predicate=predicate, arguments=arguments, misc=kwargs)


class Goal(BaseModel):
    state: list[Belief]
    misc: dict[str, Any]

    def __init__(self, state: list[Belief], **kwargs) -> None:
        super().__init__(state=state, misc=kwargs)


class Observation(BaseModel):
    observation_type: str
    value: Any

    def __str__(self) -> str:
        return f'[{self.observation_type}] {self.value}'


class ActionStatus(BaseModel):
    status: Any


class ConnType:
    send = 'send'
    request = 'request'


class AgentCmd(BaseModel):
    START: ClassVar[str] = 'start'
    STOP: ClassVar[str] = 'stop'

    agent_id: str
    cmd: str
    args: dict[str, Any] = dict()


class StatusReport(BaseModel):
    CREATED: ClassVar[str] = 'CREATED'
    READY: ClassVar[str] = 'READY'
    RUNNING: ClassVar[str] = 'RUNNING'
    FINISHED: ClassVar[str] = 'FINISHED'
    ERROR: ClassVar[str] = 'ERROR'
    TIMEOUT: ClassVar[str] = 'TIMEOUT'

    agent_id: str
    module_id: str
    status: str
    ts: float
    args: dict[str, Any] = dict()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}[{self.agent_id}.{self.module_id}]({self.status}, {self.ts}{f": {self.args}" if self.args else ""})'


class MsgHeader(BaseModel):
    uuid: bytes
    sender_id: str
    recipient_id: str
    ts: float | str
    performative: str


class Message(BaseModel):
    short_uuid_format: ClassVar[bool] = True

    header: MsgHeader
    body: Any | dict[str, Any]

    def __init__(self,
                 body: Any | dict[str, Any],
                 sender_id: str = '',
                 recipient_id: str = '',
                 ts: float | str = '',
                 performative: str = '',
                 header: MsgHeader | None = None) -> None:
        if header is None:
            if not sender_id or not recipient_id or not ts:
                raise ValueError('Missing message values!')
            super().__init__(
                header=MsgHeader(
                    uuid=uuid4().bytes,
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    ts=ts,
                    performative=performative),
                body=body)
        else:
            super().__init__(header=header, body=body)

    def __str__(self) -> str:
        return f'[{self.header.ts}][{self.header.sender_id}-->{self.header.recipient_id}]' \
               f'{f"[{self.header.performative}]" if self.header.performative else ""} ' \
               f'{self.body}'

    @property
    def id(self) -> str:
        return self.short_id if self.short_uuid_format else self.full_id

    @property
    def full_id(self) -> str:
        return self.header.uuid.hex()

    @property
    def short_id(self) -> str:
        return self.header.uuid[-6:].hex()

    def dict(self, *args, **kwargs) -> dict[str, Any]:
        return {
            'header': self.header.model_dump(),
            'body': self.body
        }

    def model_dump(self, *args, **kwargs) -> dict:
        return self.dict(*args, **kwargs)


class Outbox(ABC):
    def __init__(self) -> None:
        self._msgs: dict[tuple[str, str, str | None], list[Any | dict[str, Any]]] = dict()

        self._recipients = list()
        self._next_recipient = -1
        self._next_content = -1

    def _add(self, recipient_id: str, performative: str, content: Any | dict[str, Any], extension: str = '') -> None:
        if (recipient_id, performative) in self._msgs:
            self._msgs[recipient_id, performative, extension].append(content)
        else:
            self._msgs[recipient_id, performative, extension] = [content]

    def __iter__(self) -> Self:
        self._recipients = list(self._msgs.keys())
        self._next_recipient = 0
        self._next_content = 0
        return self

    def __next__(self) -> tuple[str, str, str, Any | dict[str, Any]]:
        if self._next_content >= len(self._msgs[self._recipients[self._next_recipient]]):
            self._next_recipient += 1
            self._next_content = 0

        if self._next_recipient >= len(self._msgs):
            raise StopIteration

        recipient_id, performative, extension = self._recipients[self._next_recipient]
        content = self._msgs[self._recipients[self._next_recipient]][self._next_content]
        self._next_content += 1
        return recipient_id, performative, extension, content

    def __bool__(self) -> bool:
        return bool(self._msgs)

    def __str__(self) -> str:
        messages = list()
        for recipient, content_list in self._msgs.items():
            for content in content_list:
                messages.append(f'[TO: {recipient[0]}][{recipient[1]}{f"/{recipient[2]}" if recipient[2] else ""}]({content})')
        return f'Outbox({", ".join(messages)})'

    def __repr__(self) -> str:
        return self.__str__()


class ModuleTypes:
    AGENT = 'Agent'
    ALL = 'All'
    PERCEPTOR = 'Perceptor'
    ACTUATOR = 'Actuator'
    LLREASONER = 'LLReasoner'
    LEARNER = 'Learner'
    KNOWLEDGE = 'Knowledge'
    HLREASONER = 'HLReasoner'
    GOALGRAPH = 'GoalGraph'
    MEMORY = 'Memory'


class Directory:
    def __init__(self,
                 perception: Iterable[str] | str,
                 actuation: Iterable[str] | str,
                 ll_reasoning: Iterable[str] | str,
                 learning: Iterable[str] | str | None = None,
                 knowledge: Iterable[str] | str | None = None,
                 hl_reasoning: Iterable[str] | str | None = None,
                 goals: Iterable[str] | str | None = None,
                 memory: Iterable[str] | str | None = None
                 ) -> None:
        self._perception = self._process_modules(perception)
        self._actuation = self._process_modules(actuation)
        self._ll_reasoning = self._process_modules(ll_reasoning)
        self._learning = self._process_modules(learning)
        self._knowledge = self._process_modules(knowledge)
        self._hl_reasoning = self._process_modules(hl_reasoning)
        self._goals = self._process_modules(goals)
        self._memory = self._process_modules(memory)

    @property
    def perception(self) -> list[str]:
        return self._perception

    @property
    def actuation(self) -> list[str]:
        return self._actuation

    @property
    def ll_reasoning(self) -> list[str]:
        return self._ll_reasoning

    @property
    def learning(self) -> list[str]:
        return self._learning

    @property
    def knowledge(self) -> list[str]:
        return self._knowledge

    @property
    def hl_reasoning(self) -> list[str]:
        return self._hl_reasoning

    @property
    def goals(self) -> list[str]:
        return self._goals

    @property
    def memory(self) -> list[str]:
        return self._memory

    @staticmethod
    def _process_modules(module_ids: Iterable[str] | str | None) -> list[str]:
        if isinstance(module_ids, Iterable):
            modules = list(module_ids)
        elif isinstance(module_ids, str):
            modules = [module_ids]
        else:
            modules = []
        return modules