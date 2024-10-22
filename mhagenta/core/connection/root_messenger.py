import asyncio
import logging
from typing import Iterable, Callable

from mhagenta.utils.common import MHABase, AgentTime, ModuleTypes, DEFAULT_LOG_FORMAT
from mhagenta.core.connection.connector import Connector
from mhagenta.utils import AgentCmd, StatusReport, LoggerExtras


class RootMessenger(MHABase):
    def __init__(self,
                 connector_cls: type[Connector],
                 agent_id: str,
                 agent_time: AgentTime,
                 status_callback: Callable[[StatusReport], None],
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
        self._status_callback = status_callback

        self._connector = connector_cls(
            agent_id=agent_id,
            agent_time=agent_time,
            sender_id=f'root.{agent_id}',
            log_tags=self._log_tags,
            log_level=log_level,
            log_format=log_format,
            **kwargs
        )

    async def initialize(self) -> None:
        await self._connector.initialize()

        async with asyncio.TaskGroup() as tg:
            tasks: list[asyncio.Task] = list()
            tasks.append(tg.create_task(self._connector.register_cmd_out_channel()))
            tasks.append(tg.create_task(self._connector.subscribe_to_statuses(self._status_callback_generator(self._status_callback))))

    async def start(self):
        await self._connector.start()

    async def stop(self):
        await self._connector.stop()

    def cmd(self,
            cmd: AgentCmd,
            module_types: str | Iterable[str] = ModuleTypes.ALL,
            module_ids: str | Iterable[str] = ModuleTypes.ALL):
        self._connector.cmd(cmd, module_types, module_ids)

    def _status_callback_generator(self, status_callback: Callable[[StatusReport], None]) -> Callable[[StatusReport], None]:
        def callback(status: StatusReport) -> None:
            self.debug(f'Received {status}')
            status_callback(status)

        return callback

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )
