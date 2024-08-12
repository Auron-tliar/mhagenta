import logging
from enum import Enum

from pika.spec import Basic, BasicProperties
from pika.channel import Channel
from pika.exchange_type import ExchangeType

from mhagenta.utils import AgentConnector, StatusReport, ModuleTypes, ModuleTypes, AgentCmd
from ._rabbitmq_core import RabbitMQPersistentConnection
from typing import Callable, Iterable
import json
from threading import Lock as Lock


class RabbitMQAgentConnector(AgentConnector):
    class Status(Enum):
        created = 10
        initializing = 20
        ready = 30
        starting = 40
        running = 50
        stopping = 60
        stopped = 70

    def __init__(self,
                 agent_id: str,
                 status_callback: Callable[[StatusReport], None],
                 log_func: Callable[[int, str], None],
                 host: str = 'localhost',
                 port: int = 5672,
                 prefetch_count: int = 1
                 ):
        self._status_exchange_name = f'{agent_id}.status'
        self._status_routing_key = f'{agent_id}'
        self._cmd_exchange_name = f'{agent_id}.cmd'

        self._lock = Lock()

        super().__init__(
            agent_id=agent_id,
            status_callback=status_callback,
            log_func=log_func
        )

        self._core = RabbitMQPersistentConnection(
            owner_id=f'{agent_id}.RabbitMQAgentConnector',
            host=host,
            port=port,
            log_func=self.log,
            prefetch_count=prefetch_count,
            is_master=True
        )

        self._status = self.Status.created

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status == self.Status.running

    async def _subscribe_to_statuses(self):
        self.log(logging.DEBUG, 'Subscribing to agent statuses...')
        await self._core.add_consumer(
            exchange_name=self._status_exchange_name,
            exchange_type=ExchangeType.fanout,
            routing_keys=self._status_routing_key,
            msg_callback=self._process_status
        )

    async def _register_cmd_sender(self):
        self.log(logging.DEBUG, 'Registering agent cmd publisher...')
        await self._core.add_publisher(
            exchange_name=self._cmd_exchange_name,
            exchange_type=ExchangeType.topic
        )

    async def initialize(self):
        self._status = self.Status.initializing
        await self._core.start()
        self.log(logging.DEBUG, 'Connection to RabbitMQ server established, initializing connections...')
        await super().initialize()
        self.log(logging.DEBUG, 'Finished establishing connections!')
        self._status = self.Status.ready

    async def start(self):
        with self._lock:
            self._status = self.Status.running
            self.log(logging.DEBUG, 'Agent RabbitMQ connector started!')

    async def stop(self):
        with self._lock:
            if self._status == self.Status.running:
                self._status = self.Status.stopping
                await self._core.stop()
                self._status = self.Status.stopped

    def _process_status(self, ch: Channel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
        status = StatusReport(**json.loads(body))
        self._status_callback(status)

    def cmd(self,
            cmd: AgentCmd,
            receiver_type: ModuleTypes.LITERAL = ModuleTypes.ALL,
            module_names: str | Iterable[str] = ModuleTypes.ALL,
            module_ids: str | Iterable[str] = ModuleTypes.ALL):
        if isinstance(module_names, str):
            module_names = [module_names]
        if isinstance(module_ids, str):
            module_ids = [module_ids]
        for module_name in module_names:
            for module_id in module_ids:
                if receiver_type == ModuleTypes.ALL:
                    module_name = ModuleTypes.ALL
                if module_name == ModuleTypes.ALL:
                    module_id = ModuleTypes.ALL
                routing_key = f'{receiver_type}.{module_name}.{module_id}'
                self._core.publish_message(
                    exchange_name=self._cmd_exchange_name,
                    routing_key=routing_key,
                    message=json.dumps(cmd.model_dump()).encode('utf-8')
                )
