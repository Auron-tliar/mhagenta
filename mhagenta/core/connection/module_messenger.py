import asyncio
import logging
from typing import Iterable, Callable

from mhagenta.utils.common import MHABase, AgentTime, DEFAULT_LOG_FORMAT
from mhagenta.utils.common.typing import MsgProcessorCallback, Sender, Recipient, Channel
from mhagenta.core.connection.connector import Connector
from mhagenta.utils import Message, AgentCmd, StatusReport, LoggerExtras


class ModuleMessenger(MHABase):
    def __init__(self,
                 connector_cls: type[Connector],
                 agent_id: str,
                 module_type: str,
                 module_id: str,
                 agent_time: AgentTime,
                 out_id_channels: Iterable[tuple[Recipient, Channel]],
                 in_id_channel_callbacks: Iterable[tuple[Sender, Channel, MsgProcessorCallback]],
                 agent_cmd_callback: Callable[[AgentCmd], None],
                 log_tags: list[str],
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 **kwargs
                 ) -> None:
        super().__init__(
            agent_id=agent_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )

        self._time = agent_time
        self._out_id_channels = out_id_channels
        self._in_id_channel_callbacks = in_id_channel_callbacks
        self._agent_cmd_callback = agent_cmd_callback

        self._connector = connector_cls(
            agent_id=agent_id,
            sender_id=f'{module_type}.{module_id}',
            agent_time=self._time,
            log_tags=self._log_tags,
            log_level=log_level,
            log_format=log_format,
            **kwargs
        )

    async def initialize(self) -> None:
        await self._connector.initialize()

        async with asyncio.TaskGroup() as tg:
            tasks: list[asyncio.Task] = list()
            for recipient, channel in self._out_id_channels:
                tasks.append(tg.create_task(self._connector.register_out_channel(recipient=recipient, channel=channel)))
            for sender, channel, callback in self._in_id_channel_callbacks:
                tasks.append(tg.create_task(self._connector.subscribe_to_in_channel(sender=sender, channel=channel, callback=self._msg_callback_generator(callback))))
            tasks.append(tg.create_task(self._connector.subscribe_to_cmds(self._cmd_callback_generator(self._agent_cmd_callback))))
            tasks.append(tg.create_task(self._connector.register_status_out_channel()))

    async def start(self) -> None:
        await self._connector.start()

    async def stop(self) -> None:
        await self._connector.stop()

    def send(self, recipient: str, channel: str, msg: Message) -> None:
        self._connector.send(recipient, channel, msg)
        self.debug(f'Sent {msg}')

    def _msg_callback_generator(self, msg_callback: Callable[[Sender, Channel, Message], None]) -> Callable[[Sender, Channel, Message], None]:
        def callback(sender: str, channel: str, msg: Message) -> None:
            self.debug(f'Received {msg}')
            msg_callback(sender, channel, msg)
        return callback

    def report_status(self, status: StatusReport) -> None:
        self._connector.status(status)
        self.debug(f'Reported {status}')

    def _cmd_callback_generator(self, cmd_callback: Callable[[AgentCmd], None]) -> Callable[[AgentCmd], None]:
        def callback(cmd: AgentCmd) -> None:
            self.debug(f'Received {cmd}')
            cmd_callback(cmd)
        return callback

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )
