import logging
from typing import Any

from mhagenta.bases import ActuatorBase, PerceptorBase
from mhagenta.states import PerceptorState, ActuatorState
from mhagenta.core import RabbitMQConnector
from mhagenta.utils import Message, Performatives


class RMQReceiverBase(PerceptorBase):
    """
    Extended receiver (Perceptor) base class for inter-agent communication.

    Subclass ``on_message`` to process received dictionaries and return the updated ``PerceptorState``. The runtime
    establishes the external connection before ``on_init`` and queues incoming messages for behaviour processing.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta', **kwargs):
        """Configure the external inbox for this agent.

        Args:
            host (str, optional): RabbitMQ host reachable from the module process. Defaults to ``'localhost'``;
                the orchestrator rewrites this hostname for external modules running in containers.
            port (int, optional): RabbitMQ AMQP port. Defaults to 5672.
            exchange_name (str, optional): Exchange shared with sender modules. Defaults to ``'mhagenta'``.
            **kwargs: ``ModuleBase`` constructor arguments, including required ``module_id`` and optional
                ``initial_state``, ``init_kwargs``, and ``tags``. Supply tags as a list because this class adds its
                external communication tags to it.
        """
        super().__init__(**kwargs)
        # self._agent_id = agent_id
        self.tags.extend(['external', 'receiver', 'rmq', 'messaging'])
        self.conn_params = {
            'host': host,
            'port': port,
            'exchange_name': exchange_name
        }
        self._ext_messenger: RabbitMQConnector | None = None

    async def _internal_init(self) -> None:
        self._ext_messenger = RabbitMQConnector(
            agent_id=self._agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=self.conn_params['host'],
            port=self.conn_params['port'],
            log_tags=[self._agent_id, self.module_id, 'ExternalReceiver'],
            log_level=self._owner._log_level,
            external_exchange_name=self.conn_params['exchange_name'],
        )
        await self._ext_messenger.initialize()
        await self._ext_messenger.subscribe_to_in_channel(
            sender='',
            channel=self._agent_id,
            callback=self._on_message_callback
        )
        await self._ext_messenger.start()

    async def _internal_start(self) -> None:
        pass

    async def _internal_stop(self) -> None:
        await self._ext_messenger.stop()

    def on_message(self, state: PerceptorState, sender: str, msg: dict[str, Any]) -> PerceptorState:
        """
        Override to define the agent's reaction to receiving a message from another agent.

        Args:
            state (PerceptorState): module's internal state enriched with relevant runtime information and
                functionality.
            sender (str): sender's `agent_id`.
            msg (dict[str, Any]): message's content.

        Returns:
            PerceptorState: Updated or unchanged state for runtime processing. An override must return the state;
                this base hook is a placeholder and does not supply a default response.
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
        if self._owner._stage == self._owner.Stage.running:
            self._owner._queue.push(
                func=self._on_message_task,
                ts=self._owner.time.agent,
                priority=False,
                sender=sender,
                msg=msg
            )
        else:
            self._owner._queue.push(
                func=self._on_message_task,
                ts=self._owner.time.agent,
                priority=False,
                periodic=True,
                frequency=self._owner._control_frequency,
                stop_condition=lambda: self._owner._stage == self._owner.Stage.running,
                sender=sender,
                msg=msg
            )


class RMQSenderBase(ActuatorBase):
    """
    Extended sender (Actuator) base class for inter-agent communication.

    Call ``send()`` from an initialized behaviour hook to publish directly through the external connector.
    The recipient's receiver must use the same broker and exchange. Internal module messages still use the outbox.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta', **kwargs):
        """Configure publishing to other agents.

        Args:
            host (str, optional): RabbitMQ host reachable from the module process. Defaults to ``'localhost'``;
                the orchestrator rewrites this hostname for external modules running in containers.
            port (int, optional): RabbitMQ AMQP port. Defaults to 5672.
            exchange_name (str, optional): Exchange shared with receiver modules. Defaults to ``'mhagenta'``.
            **kwargs: ``ModuleBase`` constructor arguments, including required ``module_id`` and optional
                ``initial_state``, ``init_kwargs``, and list-valued ``tags``. External communication tags are appended.
        """
        super().__init__(**kwargs)
        # self._agent_id = agent_id
        self.tags.extend(['external', 'sender', 'rmq', 'messaging'])
        self.conn_params = {
            'host': host,
            'port': port,
            'exchange_name': exchange_name
        }
        self._ext_messenger: RabbitMQConnector | None = None

    async def _internal_init(self) -> None:
        self._ext_messenger = RabbitMQConnector(
            agent_id=self._agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=self.conn_params['host'],
            port=self.conn_params['port'],
            log_tags=[self._agent_id, 'ExternalSender'],
            log_level=self._owner._log_level,
            external_exchange_name=self.conn_params['exchange_name'],
        )
        await self._ext_messenger.initialize()
        await self._ext_messenger.register_out_channel(
            recipient='',
            channel='',
        )
        await self._ext_messenger.start()

    async def _internal_start(self) -> None:
        pass

    async def _internal_stop(self) -> None:
        await self._ext_messenger.stop()

    def send(self, recipient_id: str, msg: dict[str, Any], performative: str = Performatives.INFORM) -> None:
        """
        Call this method to send a message to another agent.

        Sender's `agent_id` is automatically added to the message's `sender` field.

        Args:
            recipient_id (Any): recipient's ID string. Typically, it can be accessed via the recipient's directory
                card (e.g. `state.directory.external[<agent_id>].address` if `agent_id` is known).
            msg (dict[str, Any]): message's content. Must be serializable (and deserializable) with `dill`.
            performative (str): message performative.
        """
        self.log(logging.DEBUG, f'Sending message to {recipient_id}.')
        msg['sender'] = self.agent_id
        self._ext_messenger.send(
            recipient=recipient_id,
            channel=recipient_id,
            msg=Message(
                body=msg,
                sender_id=self._agent_id,
                recipient_id=recipient_id,
                ts=self._owner.time.agent,
                performative=performative
            )
        )


class RMQPerceptorBase(PerceptorBase):
    """
    Extended perceptor base class for interacting with RabbitMQ-based environments.

    Call ``observe()`` from a behaviour hook to request data, then handle the response in ``on_observation()``.
    Requests and responses are dictionaries defined by the environment. To forward a result internally, construct
    an ``Observation`` and enqueue it with ``state.outbox.send_observation()`` before returning the state.
    Responses are addressed to the agent's observation route. Multiple perceptors subscribed to that route receive
    the same responses; include and echo application correlation fields when a response must be matched to a request.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta-env', **kwargs):
        """Configure observation requests and their response subscription.

        Args:
            host (str, optional): RabbitMQ host reachable from the module process. Defaults to ``'localhost'``;
                the orchestrator rewrites this hostname for external modules running in containers.
            port (int, optional): RabbitMQ AMQP port. Defaults to 5672.
            exchange_name (str, optional): External exchange shared with the environment. Defaults to
                ``'mhagenta-env'``. Set this explicitly to match the environment's configured exchange.
            **kwargs: ``ModuleBase`` constructor arguments, including required ``module_id`` and optional
                ``initial_state``, ``init_kwargs``, and list-valued ``tags``. External communication tags are appended.
        """
        super().__init__(**kwargs)
        # self._agent_id = agent_id
        self.tags.extend(['external', 'perceptor', 'rmq', 'env-perceptor'])
        self.conn_params = {
            'host': host,
            'port': port,
            'exchange_name': exchange_name
        }
        self._connector: RabbitMQConnector | None = None

    async def _internal_init(self) -> None:
        self._connector = RabbitMQConnector(
            agent_id=self._agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=self.conn_params['host'],
            port=self.conn_params['port'],
            log_tags=[self._agent_id, self.module_id],
            log_level=self._owner._log_level,
            external_exchange_name=self.conn_params['exchange_name'],
        )
        await self._connector.initialize()
        await self._connector.subscribe_to_in_channel(
            sender='',
            channel=f'{self._agent_id}::observations',
            callback=self._on_observation_callback
        )
        await self._connector.register_out_channel(
            recipient='',
            channel=''
        )
        await self._connector.start()

    async def _internal_start(self) -> None:
        pass

    async def _internal_stop(self) -> None:
        await self._connector.stop()

    def observe(self, env_id: str | None = None, **kwargs) -> None:
        """Publish an observation request and return without waiting for its response.

        Use after runtime initialization, when ``self.state`` and the external connector are available. The
        environment's response is delivered later to ``on_observation()``. Selecting an ID chooses a routing key
        on this perceptor's configured exchange; it does not switch brokers or exchanges using the directory card.

        Args:
            env_id (str, optional): Environment ID. When None, use the ``env_id`` in the first environment card's
                address. That default requires a populated external directory with at least one environment.
            **kwargs: Observation request fields passed to ``MHAEnvBase.on_observe`` as keyword arguments.
        """
        env_id = self.state.directory.external.environment.address['env_id'] if env_id is None else env_id
        self.log(logging.DEBUG, f'Sending observation request to \"{env_id}\".')
        self._connector.send(
            recipient=env_id,
            channel='',
            msg=Message(
                body=kwargs,
                sender_id=self._agent_id,
                recipient_id=env_id,
                ts=self._owner.time.agent,
                performative=Performatives.OBSERVE
            )
        )

    def on_observation(self, state: PerceptorState, env_id: str, **kwargs) -> PerceptorState:
        """
        Override to define reaction to an observation (e.g. forward it to a low-level reasoner).

        Args:
            state (PerceptorState): current perceptor state.
            env_id (str): environment id.
            **kwargs: Fields of the response dictionary returned by ``MHAEnvBase.on_observe``. Their names and
                meaning are defined by the environment; no ``Observation`` object is constructed automatically.

        Returns:
            PerceptorState: updated perceptor state.
        """
        return state

    def _on_observation_task(self, sender: str, msg: Message) -> None:
        try:
            self.log(logging.DEBUG, f'Received observation {msg.short_id} from the environment {sender}.')
            update = self.on_observation(self.state, env_id=sender, **msg.body)
            self._owner._process_update(update)
        except Exception as ex:
            self._owner.warning(
                f'Caught exception \"{ex}\" while processing observation {msg.short_id}!'
                ' Aborting processing and attempting to resume execution...')
            raise ex

    def _on_observation_callback(self, sender: str, channel: str, msg: Message):
        self._owner._queue.push(
            func=self._on_observation_task,
            ts=self._owner.time.agent,
            priority=False,
            sender=sender,
            msg=msg
        )


class RMQActuatorBase(ActuatorBase):
    """
    Extended actuator base class for interacting with RabbitMQ-based environments.

    Call ``act()`` from a behaviour hook to request an action. ``on_status()`` is invoked only when the environment
    returns a response dictionary. To report a result internally, wrap it in an ``ActionStatus`` and enqueue it
    with ``state.outbox.send_status()`` before returning the state.
    Responses are addressed to the agent's action-status route. Multiple actuators subscribed to that route receive
    the same responses; include and echo application correlation fields when a response must be matched to a request.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta-env', **kwargs):
        """Configure action requests and their status subscription.

        Args:
            host (str, optional): RabbitMQ host reachable from the module process. Defaults to ``'localhost'``;
                the orchestrator rewrites this hostname for external modules running in containers.
            port (int, optional): RabbitMQ AMQP port. Defaults to 5672.
            exchange_name (str, optional): External exchange shared with the environment. Defaults to
                ``'mhagenta-env'``. Set this explicitly to match the environment's configured exchange.
            **kwargs: ``ModuleBase`` constructor arguments, including required ``module_id`` and optional
                ``initial_state``, ``init_kwargs``, and list-valued ``tags``. External communication tags are appended.
        """
        super().__init__(**kwargs)
        # self._agent_id = agent_id
        self.tags.extend(['external', 'actuator', 'rmq', 'env-actuator'])
        self.conn_params = {
            'host': host,
            'port': port,
            'exchange_name': exchange_name
        }
        self._connector: RabbitMQConnector | None = None

    async def _internal_init(self) -> None:
        self._connector = RabbitMQConnector(
            agent_id=self._agent_id,
            sender_id=self._agent_id,
            agent_time=self._owner.time,
            host=self.conn_params['host'],
            port=self.conn_params['port'],
            log_tags=[self._agent_id, self.module_id],
            log_level=self._owner._log_level,
            external_exchange_name=self.conn_params['exchange_name'],
        )
        await self._connector.initialize()
        await self._connector.subscribe_to_in_channel(
            sender='',
            channel=f'{self._agent_id}::act_status',
            callback=self._on_status_callback
        )
        await self._connector.register_out_channel(
            recipient='',
            channel=''
        )
        await self._connector.start()

    async def _internal_start(self) -> None:
        pass

    async def _internal_stop(self) -> None:
        await self._connector.stop()

    def act(self, env_id: str | None = None, **kwargs) -> None:
        """Publish an action request and return without waiting for an action status.

        Use after runtime initialization, when ``self.state`` and the external connector are available. A status
        reaches ``on_status()`` only if ``MHAEnvBase.on_action`` returns a response dictionary. Returning state alone
        or ``(state, None)`` sends no status; ``(state, {})`` does send one. Selecting an ID does not change the
        connector's configured broker or exchange.

        Args:
            env_id (str, optional): Environment ID. When None, use the ``env_id`` in the first environment card's
                address. That default requires a populated external directory with at least one environment.
            **kwargs: Action request fields passed to ``MHAEnvBase.on_action`` as keyword arguments.
        """
        env_id = self.state.directory.external.environment.address['env_id'] if env_id is None else env_id
        self.log(logging.DEBUG, f'Sending action request to \"{env_id}\".')
        self._connector.send(
            recipient=env_id,
            channel='',
            msg=Message(
                body=kwargs,
                sender_id=self._agent_id,
                recipient_id=env_id,
                ts=self._owner.time.agent,
                performative=Performatives.ACT
            )
        )

    def on_status(self, state: ActuatorState, env_id: str, **kwargs) -> ActuatorState:
        """
        Override to define reaction to an action status (e.g. forward it to a low-level reasoner).

        Args:
            state (ActuatorState): current actuator state.
            env_id (str): environment id.
            **kwargs: Fields of the response dictionary returned by ``MHAEnvBase.on_action``. Their names and
                meaning are defined by the environment; no ``ActionStatus`` object is constructed automatically.

        Returns:
            ActuatorState: updated actuator state.
        """
        return state

    def _on_status_task(self, sender: str, msg: Message) -> None:
        try:
            self.log(logging.DEBUG, f'Received action status {msg.short_id} from the environment {sender}.')
            update = self.on_status(self.state, env_id=sender, **msg.body)
            self._owner._process_update(update)
        except Exception as ex:
            self._owner.warning(
                f'Caught exception \"{ex}\" while processing action status {msg.short_id}!'
                ' Aborting processing and attempting to resume execution...')
            raise ex

    def _on_status_callback(self, sender: str, channel: str, msg: Message) -> None:
        self._owner._queue.push(
            func=self._on_status_task,
            ts=self._owner.time.agent,
            priority=False,
            sender=sender,
            msg=msg
        )
