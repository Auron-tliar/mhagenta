import asyncio
import logging
from typing import Iterable, Any
import time

from mhagenta.utils import LoggerExtras
from mhagenta.utils.common import MHABase, DEFAULT_LOG_FORMAT, AgentTime, Message, Performatives
from mhagenta.core import RabbitMQConnector
from mhagenta.environment import MHAEnvironment


class RMQEnvironment(MHAEnvironment):
    """
    Base class for RabbitMQ-based environments
    """

    def __init__(self,
                 state: dict[str, Any] | None = None,
                 env_id: str = "environment",
                 host: str = 'localhost',
                 port: int = 5672,
                 exec_duration: float = 60.,
                 exchange_name: str = 'mhagenta-env',
                 start_time_reference: float | None = None,
                 log_id: str | None = None,
                 log_tags: list[str] | None = None,
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 tags: Iterable[str] | None = None
                 ) -> None:
        super().__init__(
            state = state,
            env_id=env_id,
            exec_duration=exec_duration,
            start_time_reference=start_time_reference,
            log_id=log_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format,
            tags=tags
        )

        self._main_task: asyncio.Task | None = None
        self._timeout_task: asyncio.Task | None = None

        self._connector = RabbitMQConnector(
            agent_id=self.id,
            sender_id=self.id,
            agent_time=self.time,
            host=host,
            port=port,
            log_tags=[self.id, 'Environment'],
            external_exchange_name=exchange_name,
        )
        self._connector.subscribe_to_in_channel(
            sender='',
            channel=self.id,
            callback=self._on_request
        )
        self._connector.register_out_channel(
            recipient='',
            channel=''
        )

    async def initialize(self) -> None:
        await self._connector.initialize()

    async def start(self) -> None:
        with asyncio.TaskGroup() as tg:
            self._main_task = tg.create_task(self._connector.start())
            tg.create_task(self._timeout())

    def stop(self) -> None:
        self._main_task.cancel()
        self._timeout_task.cancel()

    async def _timeout(self) -> None:
        await asyncio.sleep(self._exec_duration)
        self._main_task.cancel()

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

    def send_response(self, recipient_id: str, channel: str, msg: Message, **kwargs) -> None:
        self._connector.send(
            recipient=recipient_id,
            channel=recipient_id,
            msg=msg
        )
