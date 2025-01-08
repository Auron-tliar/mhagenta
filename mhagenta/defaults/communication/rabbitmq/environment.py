import asyncio
import logging
from typing import Iterable, Any
import time

from mhagenta.utils import LoggerExtras
from mhagenta.utils.common import MHABase, DEFAULT_LOG_FORMAT, AgentTime, Message, Performatives
from mhagenta.core import RabbitMQConnector


class RMQEnvironment(MHABase):
    """
    Base class for RabbitMQ-based environments
    """

    def __init__(self,
                 state: dict[str, Any],
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
            agent_id=env_id,
            log_id=log_id,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )

        self.id = env_id
        self.tags = list(tags)
        if start_time_reference is None:
            start_time_reference = time.time()
        self.time = AgentTime(
            agent_start_ts=start_time_reference,
            exec_start_ts=start_time_reference
        )
        self._exec_duration = exec_duration

        self.state = state
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

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs) -> dict[str, Any]:
        """
        Override to define the effects of an action on the environment.

        Args:
            state (dict[str, Any]): state of environment
            sender_id (str): sender agent id
            **kwargs: keyword-based description of an action

        Returns:
            dict[str, Any]: modified state

        """
        return state

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
            self.state, response = self.on_observe(state=self.state, sender_id=sender, **msg.body)
            self._connector.send(
                recipient=sender,
                channel=sender,
                msg=Message(
                    body=msg.body,
                    sender_id=self.id,
                    recipient_id=sender,
                    ts=self.time.agent,
                    performative=Performatives.INFORM
                )
            )
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while processing observation request {msg.short_id} from {sender})!'
                         f' Aborting processing and attempting to resume execution...')

    def _on_action_request(self, sender: str, channel: str, msg: Message) -> None:
        try:
            self.state = self.on_action(state=self.state, sender_id=sender, **msg.body)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while processing action {msg.short_id} from {sender})!'
                         f' Aborting processing and attempting to resume execution...')

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self.time.agent,
            mod_time=self.time.module,
            exec_time=str(self.time.exec) if self.time.exec is not None else '-',
            tags=self.log_tag_str
        )
