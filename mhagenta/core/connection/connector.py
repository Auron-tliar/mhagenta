import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import dill

from mhagenta.utils.common import Message, MHABase, StatusReport, AgentCmd, ModuleTypes, AgentTime, LoggerExtras, \
    DEFAULT_LOG_FORMAT
from mhagenta.utils.common.typing import MsgProcessorCallback


class Connector(MHABase, ABC):
    def __init__(self,
                 agent_id: str,
                 sender_id: str,
                 agent_time: AgentTime,
                 log_tags: list[str] = '',
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 *args, **kwargs
                 ) -> None:
        super().__init__(
            agent_id=agent_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )
        self._id = self.__class__.__name__
        self._sender_id = sender_id
        self._time = agent_time

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def subscribe_to_in_channel(self, sender: str, channel: str, callback: MsgProcessorCallback, **kwargs) -> None:
        pass

    @abstractmethod
    async def register_out_channel(self, recipient: str, channel: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def subscribe_to_cmds(self, callback: Callable[[AgentCmd], None], **kwargs) -> None:
        pass

    @abstractmethod
    async def register_cmd_out_channel(self, **kwargs) -> None:
        pass

    @abstractmethod
    async def subscribe_to_statuses(self, callback: Callable[[StatusReport], None], **kwargs) -> None:
        pass

    @abstractmethod
    async def register_status_out_channel(self, **kwargs) -> None:
        pass

    @abstractmethod
    def send(self, recipient: str, channel: str, msg: Message, **kwargs) -> None:
        pass

    @abstractmethod
    def cmd(self,
            cmd: AgentCmd,
            module_types: str | Iterable[str] = ModuleTypes.ALL,
            module_ids: str | Iterable[str] = ModuleTypes.ALL,
            **kwargs) -> None:
        pass

    @abstractmethod
    def status(self, status: StatusReport, **kwargs) -> None:
        pass

    @staticmethod
    def encode_msg(msg: Message) -> bytes:
        return dill.dumps(msg)

    @staticmethod
    def decode_msg(msg: bytes) -> Message:
        return dill.loads(msg)

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )

