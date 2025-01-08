import logging
from typing import Any

from mhagenta.bases import ActuatorBase, PerceptorBase
from mhagenta.states import PerceptorState, ActuatorState
from mhagenta.core import RabbitMQConnector
from mhagenta.utils import Message, Performatives


class RMQReceiverBase(PerceptorBase):
    """
    Extended receiver (Perceptor) base class for inter-agent communication.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta', **kwargs):
        super().__init__(**kwargs)
        self.tags.append('external-sender')
        self._ext_messenger = RabbitMQConnector(
            agent_id=self.agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=host,
            port=port,
            log_tags=[self.agent_id, 'ExternalReceiver'],
            external_exchange_name=exchange_name,
        )
        self._ext_messenger.subscribe_to_in_channel(
            sender='',
            channel=self._agent_id,
            callback=self._on_message_callback
        )

    async def _internal_init(self) -> None:
        await self._ext_messenger.initialize()

    async def _internal_start(self) -> None:
        await self._ext_messenger.start()

    def __del__(self) -> None:
        self._ext_messenger.stop()

    def on_message(self, state: PerceptorState, sender: str, msg: dict[str, Any]) -> PerceptorState:
        """
        Override to define agent's reaction to receiving a message from another agent.

        Args:
            state (PerceptorState): module's internal state enriched with relevant runtime information and
                functionality.
            sender (str): sender's `agent_id`.
            msg (dict[str, Any]): message's content.
        """
        pass

    def _on_message_task(self, sender: str, msg: Message) -> None:
        try:
            self.log(logging.DEBUG, f'Received message {msg.short_id} from {sender}.')
            update = self.on_message(self.state, sender, msg.body)
            self._owner._process_update(update)
        except Exception as ex:
            self._owner.warning(
                f'Caught exception \"{ex}\" while processing message {msg.short_id} from {sender}!'
                f' Aborting message processing and attempting to resume execution...')
            raise ex

    def _on_message_callback(self, sender: str, channel: str, msg: Message) -> None:
        self._owner._queue.push(
            func=self._on_message_task,
            ts=self._owner.time.agent,
            priority=False,
            sender=sender,
            msg=msg
        )


class RMQSenderBase(ActuatorBase):
    """
    Extended sender (Actuator) base class for inter-agent communication.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta', **kwargs):
        super().__init__(**kwargs)
        self.tags.append('external-sender')
        self._ext_messenger = RabbitMQConnector(
            agent_id=self.agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=host,
            port=port,
            log_tags=[self.agent_id, 'ExternalSender'],
            external_exchange_name=exchange_name,
        )
        self._ext_messenger.register_out_channel(
            recipient='',
            channel='',
        )

    async def _internal_init(self) -> None:
        await self._ext_messenger.initialize()

    async def _internal_start(self) -> None:
        await self._ext_messenger.start()

    def __del__(self) -> None:
        self._ext_messenger.stop()

    def send(self, recipient_id: str, msg: dict[str, Any], performative: str = Performatives.INFORM) -> None:
        """
        Call this method to send a message to another agent.

        Args:
            recipient_id (Any): receiver's address object. Typically, can be accessed via the recipient's directory
                card (e.g. `state.directory.external[<agent_id>].address` if `agent_id` is known).
            msg (dict[str, Any]): message's content. Must be JSON serializable.
            performative (str): message performative.
        """
        msg['sender'] = self.agent_id
        self._ext_messenger.send(
            recipient=recipient_id,
            channel=recipient_id,
            msg=Message(
                body=msg,
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                ts=self._owner.time.agent,
                performative=performative
            )
        )


class RMQPerceptorBase(PerceptorBase):
    """
    Extended perceptor base class for interacting with RabbitMQ-based environments.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta-env', **kwargs):
        super().__init__(**kwargs)
        self.tags.append('rmq-perceptor')
        self._connector = RabbitMQConnector(
            agent_id=self.agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=host,
            port=port,
            log_tags=[self.agent_id, 'RMQPerceptor'],
            external_exchange_name=exchange_name,
        )
        self._connector.subscribe_to_in_channel(
            sender='',
            channel=self._agent_id,
            callback=self._on_observation
        )
        self._connector.register_out_channel(
            recipient='',
            channel=''
        )

    def observe(self, **kwargs) -> None:
        self._connector.send(
            recipient=self.state.directory.external.environment.address,
            channel='',
            msg=Message(
                body=kwargs,
                sender_id=self.agent_id,
                recipient_id=self.state.directory.external.environment.address,
                ts=self._owner.time.agent,
                performative=Performatives.OBSERVE
            )
        )

    def on_observation(self, state: PerceptorState, **kwargs) -> PerceptorState:
        """
        Override to define reaction to an observation (e.g. forward it to a low-level reasoner).

        Args:
            state (PerceptorState): perceptor state.
            **kwargs:

        Returns:

        """
        return state

    def _on_observation(self, sender: str, channel: str, msg: Message) -> None:
        try:
            self.log(logging.DEBUG, f'Received observation {msg.short_id} from the environment.')
            update = self.on_observation(self.state, **msg.body)
            self._owner._process_update(update)
        except Exception as ex:
            self._owner.warning(
                f'Caught exception \"{ex}\" while processing observation {msg.short_id}!'
                f' Aborting processing and attempting to resume execution...')
            raise ex


class RMQActuatorBase(ActuatorBase):
    """
    Extended actuator base class for interacting with RabbitMQ-based environments.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta-env', **kwargs):
        super().__init__(**kwargs)
        self.tags.append('rmq-actuator')
        self._connector = RabbitMQConnector(
            agent_id=self.agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=host,
            port=port,
            log_tags=[self.agent_id, 'RMQActuator'],
            external_exchange_name=exchange_name,
        )
        self._connector.register_out_channel(
            recipient='',
            channel=''
        )

    def act(self, **kwargs) -> None:
        self._connector.send(
            recipient=self.state.directory.external.environment.address,
            channel='',
            msg=Message(
                body=kwargs,
                sender_id=self.agent_id,
                recipient_id=self.state.directory.external.environment.address,
                ts=self._owner.time.agent,
                performative=Performatives.ACT
            )
        )
