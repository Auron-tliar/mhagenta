import asyncio
import logging
from os import PathLike
from types import FrameType
from typing import Iterable, Any, Literal, Callable
import time
from abc import ABC, abstractmethod
from pathlib import Path
import json
import dill
import signal

from mhagenta.utils import LoggerExtras
from mhagenta.utils.common import MHABase, DEFAULT_LOG_FORMAT, AgentTime, Message, Performatives


class MHAEnvBase:
    """
    Behaviour base-class for MHAEnvironment
    """
    def __init__(self, init_state: dict[str, Any] | None) -> None:
        self.state = init_state if init_state is not None else {}
        self._log_func: Callable[[int, str], None] | None = None

    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Override to define what environment returns when observed by agents,

        Args:
            state (dict[str, Any]): state of environment
            sender_id (str): sender agent id
            **kwargs: optional keyword parameters for observation action

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: tuple of modified state and keyword-based observation description
                response.

        """
        return state, dict()

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        """
        Override to define the effects of an action on the environment.

        Args:
            state (dict[str, Any]): state of environment
            sender_id (str): sender agent id
            **kwargs: keyword-based description of an action

        Returns:
            dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]: tuple of modified state and optional keyword-based action
            response

        """
        return state

    def log(self, level: int, message: str) -> None:
        """
        Log a message via the agent internal logging system

        Args:
            level (int): log level
            message (str): log message
        """
        self._log_func(level, message)


class MHAEnvironment(MHABase, ABC):
    """
    Base class for MHAgentA environments
    """

    def __init__(self,
                 base: MHAEnvBase,
                 env_id: str = "environment",
                 exec_duration: float = 60.,
                 start_time_reference: float | None = None,
                 save_dir: PathLike | None = None,
                 save_format: Literal['json', 'dill'] = 'json',
                 log_id: str | None = None,
                 log_tags: list[str] | None = None,
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 tags: Iterable[str] | None = None
                 ) -> None:
        super().__init__(
            agent_id=env_id,
            log_id=log_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )

        self.id = env_id
        self.tags = list(tags) if tags is not None else []
        self._save_dir = Path(save_dir) if save_dir is not None else None
        self._save_format = save_format
        if start_time_reference is None:
            start_time_reference = time.time()
        self.time = AgentTime(
            agent_start_ts=start_time_reference if start_time_reference is not None else time.time(),
            exec_start_ts=start_time_reference if start_time_reference is not None else time.time()
        )
        self._exec_duration = exec_duration

        self.base = base
        self.base._log_func = self.log
        self.state = base.state

        self._main_task_group: asyncio.TaskGroup | None = None

        signal.signal(signal.SIGINT, self.on_kill)
        signal.signal(signal.SIGTERM, self.on_kill)

    def set_start_time(self, start_time_reference: float | None) -> None:
        self.time = AgentTime(
            agent_start_ts=start_time_reference,
            exec_start_ts=start_time_reference
        )

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def on_start(self) -> None:
        pass

    async def start(self) -> None:
        async with asyncio.TaskGroup() as tg:
            self._main_task_group = tg
            tg.create_task(self.on_start())
            tg.create_task(self._timeout())

    async def _timeout(self) -> None:
        await asyncio.sleep(self._exec_duration)

        self.log(logging.INFO, f'Execution timeout, {'saving the state and ' if self._save_dir is not None else ''}exiting...')
        await self.stop()

    @abstractmethod
    async def on_stop(self):
        pass

    async def stop(self) -> None:
        self.log(logging.INFO, 'Stopping environment...')
        await self.on_stop()
        if self._save_dir is not None:
            self.log(logging.DEBUG, 'Saving environment state before exiting...')
            self.save_state()

    @abstractmethod
    def send_response(self, recipient_id: str, channel: str, msg: Message, **kwargs) -> None:
        """
        Sends response to an agent request.

        Args:
            recipient_id (str): recipient agent id
            channel (str): communication channel (if used)
            msg (dict[str, Any]): response body
            **kwargs: additional keyword arguments for the response

        Returns:

        """
        pass

    def _on_request(self, sender: str, channel: str, msg: Message) -> None:
        match msg.performative:
            case Performatives.OBSERVE:
                self._on_observation_request(sender, channel, msg)
            case Performatives.ACT:
                self._on_action_request(sender, channel, msg)
            case _:
                self.warning(f'Received unknown message request: {msg.performative}! Ignoring...')

    def _on_observation_request(self, sender: str, channel: str, msg: Message) -> None:
        try:
            self.state, response = self.base.on_observe(state=self.state, sender_id=sender, **msg.body)
            msg = Message(
                body=response,
                sender_id=self.id,
                recipient_id=sender,
                ts=self.time.agent,
                performative=Performatives.INFORM
            )
            self.send_response(recipient_id=sender, channel=channel, msg=msg)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while processing observation request {msg.short_id} from {sender})!'
                         f' Aborting processing and attempting to resume execution...')

    def _on_action_request(self, sender: str, channel: str, msg: Message) -> None:
        try:
            result = self.base.on_action(state=self.state, sender_id=sender, **msg.body)
            if isinstance(result, tuple):
                self.state, response = result
            else:
                self.state = result
                response = None
            if response is not None:
                msg = Message(
                    body=response,
                    sender_id=self.id,
                    recipient_id=sender,
                    ts=self.time.agent,
                    performative=Performatives.INFORM
                )
                self.send_response(recipient_id=sender, channel=channel, msg=msg)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while processing action {msg.short_id} from {sender})!'
                         f' Aborting processing and attempting to resume execution...')

    def on_kill(self, signum: int | signal.Signals, frame: FrameType) -> None:
        self.stop()

    def save_state(self) -> None:
        if self._save_dir is None:
            return
        path = self._save_dir
        path.mkdir(exist_ok=True)
        path /= f'{self.id}.sav'
        match self._save_format:
            case 'json':
                path = path.with_suffix('.json')
                with open(path, 'w') as f:
                    json.dump(self.state, f)
            case 'dill':
                with open(path, 'wb') as f:
                    dill.dump(self.state, f)
            case _:
                raise ValueError(f'Unsupported save format: {self._save_format}!')

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self.time.agent,
            mod_time=self.time.module,
            exec_time=str(self.time.exec) if self.time.exec is not None else '-',
            tags=self.log_tag_str
        )
