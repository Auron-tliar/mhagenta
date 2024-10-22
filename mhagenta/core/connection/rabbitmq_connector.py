import asyncio
import functools
import json
import logging
from typing import Iterable, Callable

import dill
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.channel import Channel
from pika.exceptions import ConnectionClosed, ChannelClosed
from pika.exchange_type import ExchangeType
from pika.frame import Method
from pika.spec import Basic, BasicProperties

from mhagenta.utils.common.classes import ModuleTypes, AgentTime, MHABase, DEFAULT_LOG_FORMAT
from mhagenta.utils import LoggerExtras, StatusReport, AgentCmd, Message
from mhagenta.utils.common.typing import MsgProcessorCallback
from mhagenta.core.connection.connector import Connector


class RabbitMQAsyncioConsumer(MHABase):
    def __init__(self,
                 agent_id: str,
                 channel: Channel,
                 exchange_name: str,
                 exchange_type: ExchangeType,
                 routing_keys: str | Iterable[str],
                 msg_callback: Callable[[Channel, Basic.Deliver, BasicProperties, bytes], None],
                 agent_time: AgentTime,
                 prefetch_count: int = 1,
                 log_tags: list[str] = '',
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT
                 ) -> None:
        super().__init__(
            agent_id=agent_id,
            log_id='MsgConsumer',
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )

        self._exchange = exchange_name
        self._exchange_type = exchange_type
        self._queue = ''
        self._routing_keys = [routing_keys] if isinstance(routing_keys, str) else routing_keys
        self._msg_callback = msg_callback

        self._channel = channel
        self._closing = False
        self._consumer_tag = None
        self._prefetch_count = prefetch_count

        self._time = agent_time

        self._bound_queues = 0

        self._consuming = False
        self._started = False
        self._stopped = False

    async def initialize(self) -> None:
        self._setup_exchange()
        await self._wait_for_initialization()

    def _setup_exchange(self) -> None:
        self.log(logging.DEBUG, f'Declaring exchange: {self._exchange}')
        cb = functools.partial(
            self._on_exchange_declareok, userdata=self._exchange)
        self._channel.exchange_declare(
            exchange=self._exchange,
            exchange_type=self._exchange_type,
            callback=cb)

    def _on_exchange_declareok(self, frame: Method, userdata: str) -> None:
        self.log(logging.DEBUG, f'Exchange declared: {userdata}')
        self._setup_queue()

    def _setup_queue(self) -> None:
        self.log(logging.DEBUG, f'Declaring queue...')
        self._channel.queue_declare(
            queue='',
            callback=self._on_queue_declareok,
            exclusive=True
        )

    def _on_queue_declareok(self, frame: Method) -> None:
        self._queue = frame.method.queue
        self.log(logging.DEBUG, f'Binding {self._exchange} to {self._queue} with {self._routing_keys}')
        for routing_key in self._routing_keys:
            self._channel.queue_bind(
                self._queue,
                self._exchange,
                routing_key=routing_key,
                callback=self._on_bindok)

    def _on_bindok(self, frame: Method) -> None:
        done = False

        self._bound_queues += 1
        self.log(logging.DEBUG, f'Bound {self._bound_queues} routing keys of {len(self._routing_keys)}')
        if self._bound_queues == len(self._routing_keys):
            done = True

        if done:
            self._set_qos()

    def _set_qos(self) -> None:
        self._channel.basic_qos(
            prefetch_count=self._prefetch_count, callback=self._on_basic_qos_ok)

    def _on_basic_qos_ok(self, frame: Method) -> None:
        self.log(logging.DEBUG, f'QOS set to: {self._prefetch_count}')
        self._start_consuming()

    def _start_consuming(self) -> None:
        self.log(logging.DEBUG, 'Issuing consumer related RPC commands')
        self._add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            self._queue, self._on_message)
        self._consuming = True
        self._started = True
        self.log(logging.DEBUG, 'Started!')

    def _add_on_cancel_callback(self) -> None:
        self.log(logging.DEBUG, 'Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self._on_consumer_cancelled)

    def _on_consumer_cancelled(self, frame: Method) -> None:
        self.log(logging.DEBUG, f'Consumer was cancelled remotely, shutting down: {frame}')
        # if self._channel and not self._channel.is_closing:
        #     self._channel.close()

    def _on_message(self, ch: Channel, basic_deliver: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
        self._msg_callback(ch, basic_deliver, properties, body)
        self._acknowledge_message(basic_deliver.delivery_tag)

    def _acknowledge_message(self, delivery_tag: int) -> None:
        self.log(logging.DEBUG, f'Acknowledging message {delivery_tag}')
        self._channel.basic_ack(delivery_tag)

    def _stop_consuming(self) -> None:
        if self._channel and not self._channel.is_closing and not self._channel.is_closed:
            self.log(logging.DEBUG, 'Sending a Basic.Cancel RPC command to RabbitMQ')
            self._channel.queue_purge(self._queue, callback=self._on_queue_purged)

    def _on_queue_purged(self, frame: Method) -> None:
        self._channel.queue_delete(self._queue, callback=self._on_queue_deleted)

    def _on_queue_deleted(self, frame: Method) -> None:
        cb = functools.partial(
            self._on_cancelok, userdata=self._consumer_tag)
        self._channel.basic_cancel(self._consumer_tag, cb)

    def _on_cancelok(self, frame: Method, userdata: str) -> None:
        self._consuming = False
        self.log(logging.DEBUG, f'RabbitMQ acknowledged the cancellation of the consumer: {userdata}')
        self._stopped = True

    async def _wait_for_initialization(self) -> None:
        while not self._started:
            await asyncio.sleep(1.)

    async def _wait_for_stop(self) -> None:
        while not self._stopped and self._consuming:
            await asyncio.sleep(1.)

    async def start(self) -> None:
        await self.initialize()

    async def stop(self) -> None:
        if not self._stopped and not self._closing:
            self._closing = True
            self.log(logging.DEBUG, 'Stopping a consumer...')
            if self._consuming:
                self._stop_consuming()
                await self._wait_for_stop()
            self.log(logging.DEBUG, 'Consumer stopped!')

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )


class RabbitMQAsyncioPublisher(MHABase):
    def __init__(self,
                 agent_id: str,
                 sender_id: str,
                 channel: Channel,
                 exchange_name: str,
                 exchange_type: ExchangeType,
                 agent_time: AgentTime,
                 routing_key: str | None = None,
                 log_tags: list[str] = '',
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT
                 ) -> None:
        super().__init__(
            agent_id=agent_id,
            log_id='MsgPublisher',
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )
        self._publisher_id = f'{agent_id}.{sender_id}.RabbitMQConnector'

        self._channel = channel

        self._exchange = exchange_name
        self._exchange_type = exchange_type
        self._routing_key = routing_key if routing_key is not None else ''

        self._time = agent_time

        self._started = False
        self._stopping = False

    async def initialize(self) -> None:
        self._setup_exchange()
        await self._wait_for_initialization()

    def _setup_exchange(self) -> None:
        self.log(logging.DEBUG, f'Declaring exchange {self._exchange}', )
        cb = functools.partial(self._on_exchange_declareok,
                               userdata=self._exchange)
        self._channel.exchange_declare(exchange=self._exchange,
                                       exchange_type=self._exchange_type,
                                       callback=cb)

    def _on_exchange_declareok(self, frame: Method, userdata: str) -> None:
        self.log(logging.DEBUG, f'Exchange declared: {userdata}')
        self._start_publishing()

    def _start_publishing(self) -> None:
        self.log(logging.DEBUG, 'Issuing consumer related RPC commands')
        self._started = True
        self.log(logging.DEBUG, 'Started!')

    def publish_message(self, message: dict | str | bytes, routing_key: str | None = None) -> None:
        if self._channel is None or not self._channel.is_open:
            return

        properties = pika.BasicProperties(app_id=self._publisher_id,
                                          content_type='application/json')

        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if not isinstance(message, bytes):
            message = bytes(message, encoding='utf-8')

        self._channel.basic_publish(self._exchange,
                                    routing_key if routing_key is not None else self._routing_key,
                                    message,
                                    properties)
        self.log(logging.DEBUG, f'Published message!')

    async def _wait_for_initialization(self) -> None:
        while not self._started:
            await asyncio.sleep(1.)

    async def start(self) -> None:
        await self.initialize()

    async def stop(self) -> None:
        if not self._stopping:
            self.log(logging.DEBUG, 'Stopping a publisher...')
            self._stopping = True
            self.log(logging.DEBUG, 'Publisher stopped!')

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )


class RabbitMQConnector(Connector):
    def __init__(self,
                 agent_id: str,
                 sender_id: str,
                 agent_time: AgentTime,
                 host: str = 'localhost',
                 port: int = 5672,
                 prefetch_count: int = 1,
                 is_master: bool = False,
                 log_tags: list[str] = '',
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT
                 ) -> None:
        super().__init__(
            agent_id=agent_id,
            sender_id=sender_id,
            agent_time=agent_time,
            log_tags=log_tags,
            log_level=log_level,
            log_format=log_format
        )

        self._main_exchange = f'{agent_id}.main'
        self._cmd_exchange = f'{agent_id}.cmd'
        self._status_exchange = f'{agent_id}.status'

        self._reconnect_delay = 0

        self._host = host
        self._port = port
        self.connection: AsyncioConnection | None = None
        self.channel: Channel | None = None
        self._prefetch_count = prefetch_count
        self._is_master = is_master

        self._consumers: list[RabbitMQAsyncioConsumer] = list()
        self._publishers: dict[tuple[str, str | None], RabbitMQAsyncioPublisher] = dict()

        self._initialized = False
        self._started = False

    async def initialize(self) -> None:
        self.connection = self._connect()
        await self._wait_for_channel_open()
        self._initialized = True

    async def start(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def stop(self) -> None:
        if self.channel.is_closing or self.channel.is_closed or self.connection.is_closing or self.connection.is_closed:
            return

        for consumer in self._consumers:
            await consumer.stop()

        self.log(logging.DEBUG, 'All consumers stopped')

        for publisher in self._publishers.values():
            await publisher.stop()

        self.log(logging.DEBUG, 'All publishers stopped')

        if self._is_master:
            self.channel.close(0, 'Normal shutdown.')

    async def subscribe_to_in_channel(self, sender: str, channel: str, callback: MsgProcessorCallback,
                                      **kwargs) -> None:
        consumer = RabbitMQAsyncioConsumer(
            agent_id=self._agent_id,
            channel=self.channel,
            exchange_name=self._main_exchange,
            exchange_type=ExchangeType.direct,
            routing_keys=channel,
            msg_callback=self._to_pika_msg_callback(sender, channel, callback),
            agent_time=self._time,
            prefetch_count=self._prefetch_count,
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await consumer.start()
        self._consumers.append(consumer)

    async def register_out_channel(self, recipient: str, channel: str, **kwargs) -> None:
        publisher = RabbitMQAsyncioPublisher(
            agent_id=self._agent_id,
            sender_id=self._sender_id,
            channel=self.channel,
            exchange_name=self._main_exchange,
            exchange_type=ExchangeType.direct,
            agent_time=self._time,
            routing_key=channel,
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await publisher.start()
        self._publishers[(self._main_exchange, channel)] = publisher

    async def subscribe_to_cmds(self, callback: Callable[[AgentCmd], None], **kwargs) -> None:
        # self.debug('Subscribing to cmds...')
        consumer = RabbitMQAsyncioConsumer(
            agent_id=self._agent_id,
            channel=self.channel,
            exchange_name=self._cmd_exchange,
            exchange_type=ExchangeType.fanout,
            routing_keys='',
            msg_callback=self._to_pika_cmd_callback(callback),
            agent_time=self._time,
            prefetch_count=self._prefetch_count,
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await consumer.start()
        self._consumers.append(consumer)
        # self.debug('Done subscribing to cmds!')

    async def register_cmd_out_channel(self, **kwargs) -> None:
        # self.debug('Registering cmd out channel...')
        publisher = RabbitMQAsyncioPublisher(
            agent_id=self._agent_id,
            sender_id='root',
            channel=self.channel,
            exchange_name=self._cmd_exchange,
            exchange_type=ExchangeType.fanout,
            agent_time=self._time,
            routing_key='',
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await publisher.start()
        self._publishers[(self._cmd_exchange, '')] = publisher
        # self.debug('Done registering cmd out channel!')

    async def subscribe_to_statuses(self, callback: Callable[[StatusReport], None], **kwargs) -> None:
        consumer = RabbitMQAsyncioConsumer(
            agent_id=self._agent_id,
            channel=self.channel,
            exchange_name=self._status_exchange,
            exchange_type=ExchangeType.direct,
            routing_keys='root',
            msg_callback=self._to_pika_status_callback(callback),
            agent_time=self._time,
            prefetch_count=self._prefetch_count,
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await consumer.start()
        self._consumers.append(consumer)

    async def register_status_out_channel(self, **kwargs) -> None:
        publisher = RabbitMQAsyncioPublisher(
            agent_id=self._agent_id,
            sender_id=self._sender_id,
            channel=self.channel,
            exchange_name=self._status_exchange,
            exchange_type=ExchangeType.direct,
            agent_time=self._time,
            routing_key='root',
            log_tags=self._log_tags,
            log_level=self._log_level,
            log_format=self._log_format
        )
        await publisher.start()
        self._publishers[(self._status_exchange, 'root')] = publisher

    def send(self, recipient: str, channel: str, msg: Message, **kwargs) -> None:
        self._publish_message(
            exchange_name=self._main_exchange,
            routing_key=channel,
            message=self.encode_msg(msg)
        )

    def cmd(self, cmd: AgentCmd, module_types: str | Iterable[str] = ModuleTypes.ALL,
            module_ids: str | Iterable[str] = ModuleTypes.ALL, **kwargs) -> None:
        self._publish_message(
            exchange_name=self._cmd_exchange,
            routing_key='',
            message=dill.dumps(cmd)
        )

    def status(self, status: StatusReport, **kwargs) -> None:
        self._publish_message(
            exchange_name=self._status_exchange,
            routing_key='root',
            message=dill.dumps(status)
        )

    def _connect(self) -> AsyncioConnection:
        self.log(logging.DEBUG, f'Connecting to {self._host}:{self._port}...')
        self._time.sleep(self._reconnect_delay)
        if self._reconnect_delay < 30.:
            self._reconnect_delay += 1
        return AsyncioConnection(
            parameters=pika.ConnectionParameters(host=self._host, port=self._port),
            on_open_callback=self._on_connection_open,
            on_open_error_callback=self._on_connection_open_error,
            on_close_callback=self._on_connection_closed)

    def _on_connection_open(self, connection: AsyncioConnection) -> None:
        self.log(logging.DEBUG, 'Connection opened')
        self._open_channel()

    def _on_connection_open_error(self, connection: AsyncioConnection, err: BaseException) -> None:
        self.log(logging.WARNING, f'Opening connection failed: {err}! Retrying...')
        self.connection = self._connect()

    def _on_connection_closed(self, connection: AsyncioConnection, reason: BaseException) -> None:
        if isinstance(reason, ConnectionClosed) and reason.reply_code == 0:
            self.log(logging.DEBUG, f'Connection is closed by normal shutdown.')
        else:
            self.log(logging.WARNING, f'Connection unexpectedly closed, reconnecting. Reason: {reason}.')
            self._started = False
            self.connection = None
            self.channel = None
            self.connection = self._connect()

    def _open_channel(self) -> None:
        self.log(logging.DEBUG, 'Opening a new channel')
        self.connection.channel(on_open_callback=self._on_channel_open)

    async def _wait_for_channel_open(self) -> None:
        while not self._started:
            await asyncio.sleep(1.)

    def _on_channel_open(self, channel: Channel) -> None:
        self.log(logging.DEBUG, 'Channel opened')
        self.channel = channel
        self._add_on_channel_close_callback()
        self._reconnect_delay = 0.
        self._started = True

    def _add_on_channel_close_callback(self) -> None:
        self.log(logging.DEBUG, 'Adding channel close callback')
        self.channel.add_on_close_callback(self._on_channel_closed)

    def _on_channel_closed(self, channel: Channel, reason: Exception) -> None:
        if isinstance(reason, ChannelClosed) and reason.reply_code == 0:
            self.log(logging.DEBUG, f'Channel is closed by normal shutdown.')
            if not self.connection.is_closing and not self.connection.is_closed:
                self.connection.close(reply_code=0, reply_text='Normal shutdown')
        else:
            self.log(logging.WARNING, f'Channel {channel} unexpectedly closed, reconnecting. Reason {reason}.')
            self._started = False
            self.connection = None
            self.channel = None
            self.connection = self._connect()

    def _to_pika_msg_callback(self, sender: str, channel: str, callback: MsgProcessorCallback) ->\
            Callable[[pika.channel.Channel, Basic.Deliver, BasicProperties, bytes], None]:
        def pika_callback(pika_channel: pika.channel.Channel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
            message = self.decode_msg(body)
            callback(sender, channel, message)
        return pika_callback

    @staticmethod
    def _to_pika_cmd_callback(callback: Callable[[AgentCmd], None]) ->\
            Callable[[pika.channel.Channel, Basic.Deliver, BasicProperties, bytes], None]:
        def pika_callback(pika_channel: pika.channel.Channel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
            cmd: AgentCmd = dill.loads(body)
            callback(cmd)
        return pika_callback

    @staticmethod
    def _to_pika_status_callback(callback: Callable[[StatusReport], None]) ->\
            Callable[[pika.channel.Channel, Basic.Deliver, BasicProperties, bytes], None]:
        def pika_callback(pika_channel: pika.channel.Channel, method: Basic.Deliver, properties: BasicProperties, body: bytes) -> None:
            status: StatusReport = dill.loads(body)
            callback(status)
        return pika_callback

    def _publish_message(self,
                         exchange_name: str,
                         routing_key: str | None,
                         message: dict | str | bytes) -> None:
        publisher_key = (exchange_name, None) if (exchange_name, None) in self._publishers else (exchange_name, routing_key)
        self._publishers[publisher_key].publish_message(message, routing_key)

    # @property
    # def _logger_extras(self) -> LoggerExtras | None:
    #     return LoggerExtras(
    #         agent_time=self._time.agent,
    #         mod_time=self._time.module,
    #         exec_time=str(self._time.exec) if self._time.exec is not None else '-',
    #         tags=self.log_tag_str
    #     )