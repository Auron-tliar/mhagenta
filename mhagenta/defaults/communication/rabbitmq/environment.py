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

    Handles ``OBSERVE`` and ``ACT`` requests addressed to ``env_id`` on a shared external exchange. Responses are
    routed to ``<agent_id>::observations`` or ``<agent_id>::act_status``. Configure the same broker and exchange
    on the participating ``RMQPerceptorBase`` and ``RMQActuatorBase`` instances.
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
        """Configure an environment with a RabbitMQ transport.

        When using ``Orchestrator.add_environment()``, the orchestrator constructs this runtime in its container.
        Direct callers must await ``initialize()`` and then ``start()``.

        Args:
            base (MHAEnvBase): Synchronous environment behaviour and initial state.
            env_id (str, optional): Environment ID and request routing key. Defaults to ``'environment'``.
            host (str, optional): RabbitMQ host as reachable from the environment process. Defaults to ``'localhost'``.
            port (int, optional): RabbitMQ AMQP port. Defaults to 5672.
            exec_duration (float, optional): Runtime timeout in seconds after ``start()``. Defaults to 60.
            exchange_name (str, optional): Shared external exchange. Defaults to ``'mhagenta'``; supply an explicit
                matching value on the perceptors and actuators, whose default exchange is ``'mhagenta-env'``.
            start_time_reference (float, optional): Unix timestamp used as the environment clock origin.
                Defaults to construction time.
            save_dir (os.PathLike, optional): Final snapshot directory. Defaults to None, disabling persistence.
                Path strings are also accepted by the underlying runtime.
            save_format (Literal['json', 'dill'], optional): Snapshot serializer. Defaults to ``'json'``.
            log_id (str, optional): Runtime logging identifier. Defaults to the runtime class name.
            log_tags (list[str], optional): Initial runtime logging tags. Defaults to the environment ID.
            log_level (int | str, optional): Runtime and connector logging threshold. Defaults to ``logging.DEBUG``.
            log_format (str, optional): Runtime logging format. Defaults to ``DEFAULT_LOG_FORMAT``.
            tags (Iterable[str], optional): Environment tags for directory searches. Defaults to no tags.
        """
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
