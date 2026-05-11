import logging
import os
from typing import Literal
from collections.abc import Iterable
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
                 save_dir: os.PathLike | None = None,
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

        self._registered_out_keys = set()
        self._tmp_pending_out: dict[str, list[Message]] = {}

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
        routing_key = f'{recipient_id}::{channel}'
        if routing_key not in self._registered_out_keys:
            if routing_key not in self._tmp_pending_out:
                assert self._main_task_group is not None, 'Main task group not set'
                self._tmp_pending_out[routing_key] = [msg]
                # self._main_task_group.create_task(self._connector.register_out_channel('', routing_key))
                self._main_task_group.create_task(self._register_out_and_send(routing_key))
            else:
                self._tmp_pending_out[routing_key].append(msg)
            return

        self._connector.send(
            recipient=f'{recipient_id}::{channel}',
            channel='',
            msg=msg
        )

    async def _register_out_and_send(self,  routing_key: str) -> None:
        await self._connector.register_out_channel('', routing_key)
        self._registered_out_keys.add(routing_key)

        for msg in self._tmp_pending_out.get(routing_key, []):
            self._connector.send(
                recipient=routing_key,
                channel='',
                msg=msg
            )
        del self._tmp_pending_out[routing_key]
