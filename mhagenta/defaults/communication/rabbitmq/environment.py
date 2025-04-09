import asyncio
import logging
from os import PathLike
from typing import Iterable, Literal
from mhagenta.utils.common import DEFAULT_LOG_FORMAT, Message
from mhagenta.core import RabbitMQConnector
from mhagenta.environment import MHAEnvironment, MHAEnvBase


class RMQEnvironment(MHAEnvironment):
    """
    RabbitMQ-based environment
    """

    def __init__(self,
                 base: MHAEnvBase,
                 env_id: str = "environment",
                 host: str = 'localhost',
                 port: int = 5672,
                 exec_duration: float = 60.,
                 exchange_name: str = 'mhagenta',
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
            base=base,
            env_id=env_id,
            exec_duration=exec_duration,
            start_time_reference=start_time_reference,
            save_dir=save_dir,
            save_format=save_format,
            log_id=log_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format,
            tags=tags
        )

        self._connector = RabbitMQConnector(
            agent_id=self.id,
            sender_id=self.id,
            agent_time=self.time,
            host=host,
            port=port,
            log_tags=[self.id, 'Environment'],
            log_level=log_level,
            external_exchange_name=exchange_name,
        )

    async def initialize(self) -> None:
        await self._connector.initialize()
        await self._connector.subscribe_to_in_channel(
            sender='',
            channel=self.id,
            callback=self._on_request
        )
        await self._connector.register_out_channel(
            recipient='',
            channel=''
        )

    async def on_start(self) -> None:
        await self._connector.start()

    async def on_stop(self) -> None:
        await self._connector.stop()

    def send_response(self, recipient_id: str, channel: str, msg: Message, **kwargs) -> None:
        self._connector.send(
            recipient=recipient_id,
            channel='',
            msg=msg
        )
