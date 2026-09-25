import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import dill

from mhagenta.utils.common import Message, MHABase, StatusReport, AgentCmd, ModuleTypes, AgentTime, LoggerExtras, \
    DEFAULT_LOG_FORMAT
from mhagenta.utils.common.typing import MsgProcessorCallback


class Connector(MHABase, ABC):
    """Transport interface for internal module messages, root commands, and status reports.

    Supply an implementation through ``Orchestrator(connector_cls=..., connector_kwargs=...)`` or an agent-level
    override. Implement the inherited async ``initialize()`` method to establish transport resources. Messengers
    await initialization, then await channel registrations/subscriptions (which may run concurrently), and later
    call ``start()``. ``stop()`` must release resources owned by the connector.

    Registration and lifecycle methods are async; ``send``, ``cmd``, and ``status`` are synchronous entry points
    used by the scheduling loop. They should enqueue or publish promptly. Incoming callbacks are synchronous and
    return None; the module messenger/runtime handles scheduling and behaviour-state updates separately.

    Transport implementations are responsible for routing and wire serialization. The provided message codec
    uses dill and preserves the complete ``Message`` envelope. Both endpoints must use compatible codecs and have
    any Python definitions needed to deserialize payloads.
    """
    def __init__(self,
                 agent_id: str,
                 sender_id: str,
                 agent_time: AgentTime,
                 log_tags: list[str] | None = None,
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT,
                 *args, **kwargs
                 ) -> None:
        """Store shared identity, clock, and logging configuration.

        Args:
            agent_id (str): Agent whose internal communication this connector serves.
            sender_id (str): Endpoint identity, supplied as ``<module_type>.<module_id>`` by module messengers
                or ``root.<agent_id>`` by the root messenger. External connectors may use an entity ID.
            agent_time (AgentTime): Runtime clock used for logging and message timing.
            log_tags (list[str], optional): Parent logging tags. Defaults to the agent ID.
            log_level (int | str, optional): Logging threshold. Defaults to ``logging.DEBUG``.
            log_format (str, optional): Logging format. Defaults to ``DEFAULT_LOG_FORMAT``.
            *args: Extension arguments accepted but unused by this base implementation.
            **kwargs: Extension options are accepted but unused here; concrete connectors consume transport settings.
        """
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
        """Activate an initialized connector whose required routes have been registered."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop transport activity and release this connector's subscriptions and publishing resources."""
        pass

    @abstractmethod
    async def subscribe_to_in_channel(self, sender: str, channel: str, callback: MsgProcessorCallback, **kwargs) -> None:
        """Register a callback for incoming application messages.

        Args:
            sender (str): Expected sender's module ID for an internal route. External transports may derive the
                actual sender from the message envelope instead.
            channel (str): Route identifier supplied by the messenger.
            callback (MsgProcessorCallback): Synchronous ``callback(sender, channel, message) -> None`` invoked
                with a decoded ``Message``. It does not return a behaviour ``State``.
            **kwargs: Transport-specific subscription options.
        """
        pass

    @abstractmethod
    async def register_out_channel(self, recipient: str, channel: str, **kwargs) -> None:
        """Prepare a route before publishing application messages.

        Args:
            recipient (str): Intended recipient's module or external entity ID.
            channel (str): Route identifier subsequently used by ``send()``.
            **kwargs: Transport-specific publishing options.
        """
        pass

    @abstractmethod
    async def subscribe_to_cmds(self, callback: Callable[[AgentCmd], None], **kwargs) -> None:
        """Subscribe a module endpoint to agent-control commands.

        Args:
            callback (Callable[[AgentCmd], None]): Synchronous callback receiving one decoded ``AgentCmd``.
            **kwargs: Transport-specific subscription options.
        """
        pass

    @abstractmethod
    async def register_cmd_out_channel(self, **kwargs) -> None:
        """Prepare command publishing for the root endpoint.

        Args:
            **kwargs: Transport-specific publishing options.
        """
        pass

    @abstractmethod
    async def subscribe_to_statuses(self, callback: Callable[[StatusReport], None], **kwargs) -> None:
        """Subscribe the root endpoint to module status reports.

        Args:
            callback (Callable[[StatusReport], None]): Synchronous callback receiving one decoded report.
            **kwargs: Transport-specific subscription options.
        """
        pass

    @abstractmethod
    async def register_status_out_channel(self, **kwargs) -> None:
        """Prepare status publishing for a module endpoint.

        Args:
            **kwargs: Transport-specific publishing options.
        """
        pass

    @abstractmethod
    def send(self, recipient: str, channel: str, msg: Message, **kwargs) -> None:
        """Publish an application message through a registered route.

        Args:
            recipient (str): Recipient's module or external entity ID.
            channel (str): Channel registered for this route. Its interpretation is transport-specific.
            msg (Message): A complete message envelope to encode and publish.
            **kwargs: Transport-specific send options.
        """
        pass

    @abstractmethod
    def cmd(self,
            cmd: AgentCmd,
            module_types: str | Iterable[str] = ModuleTypes.ALL,
            module_ids: str | Iterable[str] = ModuleTypes.ALL,
            **kwargs) -> None:
        """Publish an agent-control command through the command route.

        Args:
            cmd (AgentCmd): Command envelope containing the target agent ID and command arguments.
            module_types (str | Iterable[str], optional): Requested module-role selector. Defaults to
                ``ModuleTypes.ALL``. Selector support depends on the transport.
            module_ids (str | Iterable[str], optional): Requested module-ID selector. Defaults to
                ``ModuleTypes.ALL``. The built-in RabbitMQ connector broadcasts commands and does not filter
                by either selector.
            **kwargs: Transport-specific send options.
        """
        pass

    @abstractmethod
    def status(self, status: StatusReport, **kwargs) -> None:
        """Publish a module status report to the root controller.

        Args:
            status (StatusReport): Report envelope, including agent and module identity.
            **kwargs: Transport-specific send options.
        """
        pass

    @staticmethod
    def encode_msg(msg: Message) -> bytes:
        """Serialize a complete message with dill.

        Args:
            msg (Message): Envelope and payload to serialize. Referenced Python objects must support dill.

        Returns:
            bytes: Wire payload understood by ``decode_msg()``. Serialization errors propagate to the caller.
        """
        return dill.dumps(msg)

    @staticmethod
    def decode_msg(msg: bytes) -> Message:
        """Deserialize a message produced by the matching dill codec.

        Args:
            msg (bytes): Encoded message envelope.

        Returns:
            Message: Decoded envelope. Deserialization errors propagate to the caller.
        """
        return dill.loads(msg)

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )

