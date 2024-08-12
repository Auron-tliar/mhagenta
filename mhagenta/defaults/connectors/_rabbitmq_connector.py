import logging
from array import array
from mmap import mmap
from enum import Enum

from pika.spec import Basic, BasicProperties
from pika.channel import Channel as PikaChannel
from pika.exchange_type import ExchangeType

from mhagenta.utils import ModuleConnector, AgentCmd, StatusReport, ModuleTypes, ModuleTypes, Message
from mhagenta.utils.common.typing import *
from ._rabbitmq_core import RabbitMQPersistentConnection
from typing import Any, Iterable, Callable, Literal
import json
import pickle
import dill
from threading import Lock as Lock


class RabbitMQConnector(ModuleConnector):
    _serializer: Literal['pickle', 'json', 'dill'] = 'pickle'
    dumps = pickle.dumps
    loads = pickle.loads

    class Status(Enum):
        created = 10
        initializing = 20
        ready = 30
        starting = 40
        running = 50
        stopping = 60
        stopped = 70

        def __cmp__(self, other):
            if not isinstance(other, self.__class__):
                raise TypeError('Status class object can only be compared with another Status object!')
            return self.value.__cmp__(other.value)

    def __init__(self,
                 agent_id: str,
                 module_name: str,
                 module_id: str,
                 out_id_channels: Iterable[tuple[Recipient, Channel]],
                 in_id_channels_callbacks: Iterable[tuple[Sender, Channel, MsgProcessorCallback]],
                 agent_cmd_callback: Callable[[AgentCmd], None],
                 log_func: Callable[[int, str], None],
                 host: str = 'localhost',
                 port: int = 5672,
                 prefetch_count: int = 1):
        self._cmd_exchange_name = f'{agent_id}.cmd'
        self._cmd_routing_keys = [
            f'{ModuleTypes.ALL}.{ModuleTypes.ALL}.{ModuleTypes.ALL}',
            f'{ModuleTypes.MODULE}.{ModuleTypes.ALL}.{ModuleTypes.ALL}',
            f'{ModuleTypes.MODULE}.{module_name}.{ModuleTypes.ALL}',
            f'{ModuleTypes.MODULE}.{module_name}.{module_id}'
        ]
        self._status_exchange_name = f'{agent_id}.status'
        self._main_exchange_name = f'{agent_id}.main'

        self._lock = Lock()

        super().__init__(
            agent_id=agent_id,
            module_name=module_name,
            module_id=module_id,
            out_id_channels=out_id_channels,
            in_id_channels_callbacks=in_id_channels_callbacks,
            agent_cmd_callback=agent_cmd_callback,
            log_func=log_func
        )

        self._core = RabbitMQPersistentConnection(
            owner_id=f'{agent_id}.{module_id}.RabbitMQConnector',
            host=host,
            port=port,
            log_func=self.log,
            prefetch_count=prefetch_count,
            is_master=False
        )

        self._status = self.Status.created

    async def initialize(self):
        self._status = self.Status.initializing
        await self._core.start()
        self.log(logging.DEBUG, 'Connection to RabbitMQ server established, initializing connections...')
        await super().initialize()
        self.log(logging.DEBUG, 'Finished establishing connections!')
        self._status = self.Status.ready

    async def _subscribe_to_agent_cmd(self):
        self.log(logging.DEBUG, 'Subscribing to agent cmds...')

        await self._core.add_consumer(
            exchange_name=self._cmd_exchange_name,
            exchange_type=ExchangeType.topic,
            routing_keys=self._cmd_routing_keys,
            msg_callback=self._process_command
        )

    async def _register_status_exchange(self):
        self.log(logging.DEBUG, 'Registering agent status publisher...')

        await self._core.add_publisher(
            exchange_name=self._status_exchange_name,
            exchange_type=ExchangeType.fanout,
        )

    async def _register_out_channel(self, receiver_id: str, channel: str):
        self.log(logging.DEBUG, f'Registering out channel {channel}')

        await self._core.add_publisher(
            exchange_name=self._main_exchange_name,
            exchange_type=ExchangeType.direct,
            routing_key=channel
        )

    async def _subscribe_to_in_channel(self, sender: str, channel: str, callback: MsgProcessorCallback) -> None:
        self.log(logging.DEBUG, f'Subscribing to in-channel {channel}')

        await self._core.add_consumer(
            exchange_name=self._main_exchange_name,
            exchange_type=ExchangeType.direct,
            routing_keys=channel,
            msg_callback=self._msg_processor_factory(sender, channel, callback)
        )

    async def start(self):
        with self._lock:
            self._status = self.Status.running
            self.log(logging.DEBUG, 'Started!')

    async def stop(self):
        with self._lock:
            if self._status == self.Status.running:
                self._status = self.Status.stopping
                await self._core.stop()
                self._status = self.Status.stopped

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status == self.Status.running

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._status >= self.Status.stopping

    def _process_command(self, ch: PikaChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
        command = AgentCmd(**json.loads(body))
        self._agent_cmd_callback(command)

    def _msg_processor_factory(self, sender: str, channel: str, callback: MsgProcessorCallback):
        def process_message(ch: PikaChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
            message = Message(**self.loads(body))
            callback(sender, channel, message)
        return process_message

    def send(self, receiver_id: str, channel: str, msg: Message) -> None:
        self.log(logging.DEBUG, f'Sending message to {channel} routing_key')
        self._core.publish_message(
            exchange_name=self._main_exchange_name,
            routing_key=channel,
            message=self.dumps(msg.dict())
        )

    def report_status(self, status: StatusReport):
        self._core.publish_message(
            exchange_name=self._status_exchange_name,
            routing_key=None,
            message=status.model_dump()
        )

    @classmethod
    def set_serializer(cls, serializer: Literal['pickle', 'json', 'dill']):
        if cls._serializer == serializer:
            return

        if serializer == 'pickle':
            cls.dumps = pickle.dumps
            cls.loads = pickle.loads
        elif serializer == 'json':
            def json_dumps(*args, **kwargs) -> bytes:
                return json.dumps(*args, **kwargs).encode()

            def json_loads(data: bytes | bytearray | memoryview | array | mmap, *args, **kwargs) -> Any:
                return json.loads(data.decode(), *args, **kwargs)

            cls.dumps = json_dumps
            cls.loads = json_loads
        elif serializer == 'dill':
            cls.dumps = dill.dumps
            cls.loads = dill.loads
        else:
            try:
                serializer_module = __import__(serializer)
                cls.dumps = serializer_module.dumps
                cls.loads = serializer_module.loads
            except ModuleNotFoundError as ex:
                raise ModuleNotFoundError(f'Failed to load \'{serializer}\' package!', *ex.args)

        cls._serializer = serializer
