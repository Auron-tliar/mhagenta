import logging
import sys
import os
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Callable, Self, Iterable, Optional
from uuid import uuid4
import dataclasses

from pydantic.dataclasses import dataclass

from mhagenta.utils.common.logging import ILogging
from mhagenta.utils.common.logging import DEFAULT_FORMAT as DEFAULT_LOG_FORMAT


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
    """Utility class for getting timestamps for agent components.

    """
    def __init__(self, agent_start_ts: float, exec_start_ts: float | None = None, decimals: int = 4) -> None:
        self._agent_start_ts = agent_start_ts
        self._module_start_ts: float = time.time()
        self._exec_start_ts: float | None = exec_start_ts
        self._decimals = decimals

    @property
    def system(self) -> float:
        """System time in seconds.

        Returns:
            float: System time (in seconds) as provided by `time` module.

        """
        return round(time.time(), self._decimals)

    @property
    def agent(self) -> float:
        """Agent time in seconds.

        Returns:
            float: Seconds since the agent was created (i.e. since the initialization of agent's root controller).

        """
        return round(time.time() - self._agent_start_ts, self._decimals)

    @property
    def module(self) -> float:
        """Module time in seconds.

        Returns:
            float: Seconds since the initializations of the current module.
        """
        return round(time.time() - self._module_start_ts, self._decimals)

    @property
    def exec(self) -> float | None:
        """Execution time in seconds.

        Returns:
            float | None: Seconds since the synchronous start of agent execution. Is None if the module doesn't have
            the information yet, or negative if the execution start is scheduled in the future.
         """
        return round(time.time() - self._exec_start_ts, self._decimals) if self._exec_start_ts is not None else None

    @property
    def agent_start_ts(self) -> float:
        """Unix timestamp in seconds of when the synchronous agents execution start.

        Returns:
            float: Unix timestamp in seconds.

        """
        return self._agent_start_ts

    @property
    def exec_start_ts(self) -> float | None:
        """Unix timestamp in seconds of when this agent starts execution.

        Returns:
            float | None: Unix timestamp in seconds. None if not yet defined.

        """
        return self._exec_start_ts

    def set_exec_start_ts(self, exec_start_ts: float) -> None:
        """
        Pass the information on when this agent execution is scheduled to start.

        Args:
            exec_start_ts: system-level Unix timestamp (in seconds) of agent's scheduled start time.
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
        """Full time information as a tuple.

        Returns:
            tuple[float, float, float, float | None]: tuple of all four types of timestamps in order: system, agent,
                module, execution

        """
        return self.system, self.agent, self.module, self.exec


class TagCard[T](ABC):
    def __init__(self, tags: Iterable[T]) -> None:
        self.tags: set[T] = set(tags)

    def match(self, query: Iterable[T]) -> bool:
        for tag in query:
            if tag not in self.tags:
                return False
        return True

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    def __eq__(self, other: str) -> bool:
        return self.id == other



class ICard(TagCard[str]):
    def __init__(self, module_id: str, module_type: str, tags: Iterable[str] | None = None) -> None:
        self.module_id = module_id
        self.module_type = module_type
        super().__init__(tags if tags is not None else tuple())

    @property
    def id(self) -> str:
        return self.module_id


class ECard(TagCard[str]):
    def __init__(self, agent_id: str, address: dict[str, Any], tags: Iterable[str] | None = None) -> None:
        self.agent_id = agent_id
        self.address = address
        super().__init__(tags if tags is not None else tuple())

    @property
    def id(self) -> str:
        return self.agent_id


class BaseDirectory:
    def __init__(self, content: Iterable[TagCard[str]] | None = None) -> None:
        self._content = list(content) if content is not None else list()
        self._by_id = {card.id: card for card in content} if content is not None else dict()

    def __getitem__(self, item: str | int) -> TagCard[str]:
        if isinstance(item, int):
            return self._content[item]
        else:
            return self._by_id[item]

    def __contains__(self, item: str) -> bool:
        return item in self._by_id

    def search(self, tags: Iterable[str]) -> list[TagCard[str]]:
        output = list()
        for card in self._content:
            if card.match(tags):
                output.append(card)
        return output

    def _add(self, card: TagCard[str]) -> None:
        self._content.append(card)
        self._by_id[card.id] = card


class IDirectory(BaseDirectory):
    """Directory of all module names, types, and tags for easier communication definition.

    Created by the agent's root controller.

    """
    def __init__(self) -> None:
        super().__init__()
        self._by_type: dict[str, list[ICard]] = {
            ModuleTypes.PERCEPTOR: list(),
            ModuleTypes.ACTUATOR: list(),
            ModuleTypes.LLREASONER: list(),
            ModuleTypes.LEARNER: list(),
            ModuleTypes.KNOWLEDGE: list(),
            ModuleTypes.HLREASONER: list(),
            ModuleTypes.GOALGRAPH: list(),
            ModuleTypes.MEMORY: list()
        }

    @property
    def perception(self) -> list[ICard]:
        """(Property) List of Perceptor info cards.

        Returns:
            list[ICard]: List of `module_id`s of all the agent `Perceptor`s

        """
        return self._by_type[ModuleTypes.PERCEPTOR]

    @property
    def actuation(self) -> list[ICard]:
        """(Property) List of Actuator info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `Actuator`s

        """
        return self._by_type[ModuleTypes.ACTUATOR]

    @property
    def ll_reasoning(self) -> list[ICard]:
        """(Property) List of Low-level reasoner info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `LLReasoner`s

        """
        return self._by_type[ModuleTypes.LLREASONER]

    @property
    def learning(self) -> list[ICard]:
        """(Property) List of Learner info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `Learner`s

        """
        return self._by_type[ModuleTypes.LEARNER]

    @property
    def knowledge(self) -> list[ICard]:
        """(Property) List of Knowledge model info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `Knowledge`s

        """
        return self._by_type[ModuleTypes.KNOWLEDGE]

    @property
    def hl_reasoning(self) -> list[ICard]:
        """(Property) List of High-level reasoner info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `HLReasoner`s

        """
        return self._by_type[ModuleTypes.HLREASONER]

    @property
    def goals(self) -> list[ICard]:
        """(Property) List of Goal graph info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `goal_graph`s

        """
        return self._by_type[ModuleTypes.GOALGRAPH]

    @property
    def memory(self) -> list[ICard]:
        """(Property) List of Memory structure info cards.

        Returns:
            list[ICard]: List of info cards containing `module_id`s and tags of all the agent `Memory`s

        """
        return self._by_type[ModuleTypes.MEMORY]

    def _add_module(self, module_id: str, module_type: str, tags: Iterable[str] | None = None) -> ICard:
        card = ICard(module_id, module_type, tags)
        self._add(card)
        self._by_type[module_type].append(card)
        return card

    def __getitem__(self, item: str | int) -> ICard:
        return super().__getitem__(item)

    def search(self, tags: Iterable[str]) -> list[ICard]:
        return super().search(tags)


class EDirectory(BaseDirectory):
    ENVIRONMENT = 'environment'
    localhost_win = 'http://host.docker.internal'
    localhost_linux = 'http://172.17.0.1'


    def __init__(self,
                 env_address: dict[str, Any] | None = None,
                 env_tags: Iterable[str] | None = None
                 ) -> None:
        if env_address is None:
            super().__init__()
            return

        tags = [self.ENVIRONMENT]
        if env_tags is not None:
            tags.extend(env_tags)
        env_card = ECard(self.ENVIRONMENT, env_address, tags)
        super().__init__([env_card])

    def add_agent(self, agent_id: str, address: Any, tags: Iterable[str] | None = None) -> ECard:
        card = ECard(agent_id, address, tags)
        self._add(card)
        return card

    def __getitem__(self, item: str | int) -> ECard:
        return super().__getitem__(item)

    def search(self, tags: Iterable[str]) -> list[ECard]:
        return super().search(tags)

    @property
    def environment(self) -> ECard | None:
        return self[self.ENVIRONMENT] if self.ENVIRONMENT in self._by_id else None


class Directory:
    def __init__(self,
                 env_address: Any | None = None,
                 env_tags: Iterable[str] | None = None):
        self._internal = IDirectory()
        self._external = EDirectory(env_address=env_address, env_tags=env_tags)

    @property
    def internal(self) -> IDirectory:
        return self._internal

    @property
    def external(self) -> EDirectory:
        return self._external


@dataclass
class Belief:
    """Pydantic dataclass for agent's beliefs.

    Attributes:
        predicate (str): predicate name.
        arguments (Any | tuple[Any, ...]): predicate's arguments.
        extras (dict[str, Any], optional): keyword dictionary of additional relevant information.

    """
    predicate: str
    arguments: Any | tuple[Any, ...]
    extras: Optional[dict[str, Any]] = None


@dataclass
class Goal:
    """Pydantic dataclass for agent's goals.

    Attributes:
        state (list[Belief]): Belief-based description of the goal state.
        extras (dict[str, Any]): keyword dictionary of additional relevant information.

    """
    state: list[Belief]
    extras: Optional[dict[str, Any]] = None

    def __init__(self, state: list[Belief], **kwargs) -> None:
        super().__init__(state=state, misc=kwargs)


@dataclass
class Observation:
    """Pydantic dataclass for agent's observations.

    Attributes:
        observation_type (str): type of the observation.
        content (Any): observed object.

    """
    content: Any
    observation_type: Optional[str] = 'Any'
    value: float | None = None

    def __str__(self) -> str:
        return f'[{self.observation_type}] {self.content}'


@dataclass
class ActionStatus:
    """Pydantic dataclass for action statuses.

    Attributes:
        status (Any): execution status of the action.

    """
    status: Any


class ConnType:
    send = 'send'
    request = 'request'


@dataclass
class AgentCmd:
    START: ClassVar[str] = 'start'
    STOP: ClassVar[str] = 'stop'

    agent_id: str
    cmd: str
    args: dict[str, Any] | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.args is None:
            self.args = dict()


@dataclass
class StatusReport:
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
    args: dict[str, Any] | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.args is None:
            self.args = dict()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}[{self.agent_id}.{self.module_id}]({self.status}, {self.ts}{f": {self.args}" if self.args else ""})'


@dataclass
class Performatives:
    INFORM: ClassVar[str] = 'inform'

    AGREE: ClassVar[str] = 'agree'
    CANCEL: ClassVar[str] = 'cancel'
    CFP: ClassVar[str] = 'cfp'
    CONFIRM: ClassVar[str] = 'confirm'
    NOT_UNDERSTOOD: ClassVar[str] = 'not-understood'
    PROPOSE: ClassVar[str] = 'propose'
    QUERY: ClassVar[str] = 'query'
    REFUSE: ClassVar[str] = 'refuse'
    REJECT_PROPOSAL: ClassVar[str] = 'reject-proposal'
    REQUEST: ClassVar[str] = 'request'

    OBSERVE: ClassVar[str] = 'observe'
    ACT: ClassVar[str] = 'act'


@dataclass
class Message:

    body: Any | dict[str, Any]
    sender_id: str
    recipient_id: str
    ts: float | str
    performative: str
    uuid: bytes = dataclasses.field(default_factory=lambda: uuid.uuid4().bytes)
    short_uuid_format: bool = True

    def __init__(self,
                 body: Any | dict[str, Any],
                 sender_id: str,
                 recipient_id: str,
                 ts: float | str,
                 performative: str
                 ) -> None:
        super().__init__(
            sender_id=sender_id,
            recipient_id=recipient_id,
            ts=ts,
            performative=performative,
            body=body)
        self.uuid = uuid4().bytes

    def __str__(self) -> str:
        return f'[{self.ts}][{self.sender_id}-->{self.recipient_id}]' \
               f'{f"[{self.performative}]" if self.performative else ""} ' \
               f'{self.body}'

    @property
    def id(self) -> str:
        return self.short_id if self.short_uuid_format else self.full_id

    @property
    def full_id(self) -> str:
        return self.uuid.hex()

    @property
    def short_id(self) -> str:
        return self.uuid[-6:].hex()


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

    def clear(self) -> None:
        self._msgs.clear()
        self._recipients = list()
        self._next_recipient = -1
        self._next_content = -1


class State[T: Outbox]:
    """Container for module's internal state enriched with additional information and functionality.

    State creation is handled by the agent's root controller. Can be used to access agent time, directory of agent
    module IDs, and to send out messages to other modules.

    T resolves to a module-specific outbox class for convenient hinting when sending messages to other modules.

    """
    def __init__(self, agent_id: str, module_id: str, time_func: Callable[[], float], directory: Directory, outbox: T, **kwargs) -> None:
        self._agent_id = agent_id
        self._module_id = module_id
        self._time_func = time_func
        self._directory = directory

        self._outbox = outbox

        self._custom_fields = set(kwargs.keys())
        self.__dict__.update(kwargs)

    @property
    def agent_id(self) -> str:
        """Unique ID of the parent agent.

        Returns:
            str: agent_id specified at the agent's creation.

        """
        return self._agent_id

    @property
    def module_id(self) -> str:
        """Unique (within the scope of the parent agent) identifier of the current module.

        Returns:
            str: module_id specified in the module base object.

        """
        return self._module_id

    @property
    def time(self) -> float:
        """Agent's execution current time.

        Returns:
            float: time in seconds since the start of the agent's execution.

        """
        return self._time_func()

    @property
    def directory(self) -> Directory:
        """Agent's directory of module IDs.

        Returns:
            Directory: directory object containing information about agent's module types and IDs.

        """
        return self._directory

    @property
    def outbox(self) -> T:
        """Outbox object for this module. Use its methods to send messages to other modules.

        Returns:
            T: module-specific outbox object.

        """
        return self._outbox

    def _clear_outbox(self) -> None:
        self._outbox.clear()

    def dump(self) -> dict[str, Any]:
        return {field: self.__dict__[field] for field in self._custom_fields}

    def load(self, **kwargs) -> None:
        self._custom_fields.update(kwargs.keys())
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        return str(self.dump())

    def __repr__(self) -> str:
        return str(self.dump())
