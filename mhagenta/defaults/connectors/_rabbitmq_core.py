import functools
import logging
import time
from typing import Callable, Iterable
import asyncio
import json
from threading import RLock as Lock
import pika

from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType
from pika.spec import Basic, BasicProperties
from pika.channel import Channel
from pika.frame import Method
from pika.exceptions import ConnectionClosed, ChannelClosed


class RabbitMQAsyncioConsumer:
    def __init__(self,
                 channel: Channel,
                 exchange_name: str,
                 exchange_type: ExchangeType,
                 routing_keys: str | Iterable[str],
                 msg_callback: Callable[[Channel, Basic.Deliver, BasicProperties, bytes], None],
                 log_func: Callable[[int, str], None],
                 prefetch_count: int = 1):
        self._exchange = exchange_name
        self._exchange_type = exchange_type
        self._queue = ''
        self._routing_keys = [routing_keys] if isinstance(routing_keys, str) else routing_keys
        self._log_id = f'CONSUMER<{self._exchange}.{"|".join(self._routing_keys)}>'
        self._msg_callback = msg_callback

        self._channel = channel
        self._closing = False
        self._consumer_tag = None
        self._log = log_func
        self._prefetch_count = prefetch_count

        self._bound_queues = 0

        self._consuming = False
        self._started = False
        self._stopped = False
        self._lock = Lock()

    def log(self, level: int, message: str):
        return self._log(level, f'{self._log_id}: {message}')

    def _setup_exchange(self):
        self.log(logging.DEBUG, f'Declaring exchange: {self._exchange}')
        cb = functools.partial(
            self._on_exchange_declareok, userdata=self._exchange)
        self._channel.exchange_declare(
            exchange=self._exchange,
            exchange_type=self._exchange_type,
            callback=cb)

    def _on_exchange_declareok(self, frame: Method, userdata: str):
        self.log(logging.DEBUG, f'Exchange declared: {userdata}')
        self._setup_queue()

    def _setup_queue(self):
        self.log(logging.DEBUG, f'Declaring queue...')
        self._channel.queue_declare(
            queue='',
            callback=self._on_queue_declareok,
            exclusive=True
        )

    def _on_queue_declareok(self, frame: Method):
        self._queue = frame.method.queue
        self.log(logging.DEBUG, f'Binding {self._exchange} to {self._queue} with {self._routing_keys}')
        for routing_key in self._routing_keys:
            self._channel.queue_bind(
                self._queue,
                self._exchange,
                routing_key=routing_key,
                callback=self._on_bindok)

    def _on_bindok(self, frame: Method):
        done = False
        with self._lock:
            self._bound_queues += 1
            self.log(logging.DEBUG, f'Bound {self._bound_queues} routing keys of {len(self._routing_keys)}')
            if self._bound_queues == len(self._routing_keys):
                done = True
        if done:
            self._set_qos()

    def _set_qos(self):
        self._channel.basic_qos(
            prefetch_count=self._prefetch_count, callback=self._on_basic_qos_ok)

    def _on_basic_qos_ok(self, frame: Method):
        self.log(logging.DEBUG, f'QOS set to: {self._prefetch_count}')
        self._start_consuming()

    def _start_consuming(self):
        self.log(logging.DEBUG, 'Issuing consumer related RPC commands')
        self._add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            self._queue, self._on_message)
        with self._lock:
            self._consuming = True
            self._started = True

    def _add_on_cancel_callback(self):
        self.log(logging.DEBUG, 'Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self._on_consumer_cancelled)

    def _on_consumer_cancelled(self, frame: Method):
        self.log(logging.DEBUG, f'Consumer was cancelled remotely, shutting down: {frame}')
        # if self._channel and not self._channel.is_closing:
        #     self._channel.close()

    def _on_message(self, ch: Channel, basic_deliver: Basic.Deliver, properties: BasicProperties, body: bytes):
        self._msg_callback(ch, basic_deliver, properties, body)
        self._acknowledge_message(basic_deliver.delivery_tag)

    def _acknowledge_message(self, delivery_tag: int):
        self.log(logging.DEBUG, f'Acknowledging message {delivery_tag}')
        self._channel.basic_ack(delivery_tag)

    def _stop_consuming(self):
        if self._channel and not self._channel.is_closing and not self._channel.is_closed:
            self.log(logging.DEBUG, 'Sending a Basic.Cancel RPC command to RabbitMQ')
            self._channel.queue_purge(self._queue, callback=self._on_queue_purged)

    def _on_queue_purged(self, frame: Method):
        self._channel.queue_delete(self._queue, callback=self._on_queue_deleted)

    def _on_queue_deleted(self, frame: Method):
        cb = functools.partial(
            self._on_cancelok, userdata=self._consumer_tag)
        self._channel.basic_cancel(self._consumer_tag, cb)

    def _on_cancelok(self, frame: Method, userdata: str):
        with self._lock:
            self._consuming = False
            self.log(logging.DEBUG, f'RabbitMQ acknowledged the cancellation of the consumer: {userdata}')
            self._stopped = True

    async def _wait_for_initialization(self):
        while True:
            with self._lock:
                if self._started:
                    break
            await asyncio.sleep(1.)

    async def _wait_for_stop(self):
        while True:
            with self._lock:
                if self._stopped or not self._consuming:
                    break
            await asyncio.sleep(1.)

    async def start(self):
        self._setup_exchange()
        await self._wait_for_initialization()

    async def stop(self):
        if not self._stopped and not self._closing:
            self._closing = True
            self.log(logging.DEBUG, 'Stopping a consumer...')
            if self._consuming:
                self._stop_consuming()
                await self._wait_for_stop()
            self.log(logging.DEBUG, 'Consumer stopped!')


class RabbitMQAsyncioPublisher(object):
    def __init__(self,
                 owner_id: str,
                 channel: Channel,
                 exchange_name: str,
                 exchange_type: ExchangeType,
                 log_func: Callable[[int, str], None],
                 routing_key: str | None = None):
        self._owner_id = owner_id
        self._channel = channel

        self._exchange = exchange_name
        self._exchange_type = exchange_type
        self._routing_key = routing_key if routing_key is not None else ''
        self._log_id = f'PUBLISHER<{self._exchange}{f".{self._routing_key}" if self._routing_key else ""}>'

        self._started = False
        self._stopping = False

        self._log = log_func

        self._lock = Lock()

    def log(self, level: int, message: str):
        return self._log(level, f'{self._log_id}: {message}')

    def _setup_exchange(self):
        self.log(logging.DEBUG, f'Declaring exchange {self._exchange}', )
        cb = functools.partial(self._on_exchange_declareok,
                               userdata=self._exchange)
        self._channel.exchange_declare(exchange=self._exchange,
                                       exchange_type=self._exchange_type,
                                       callback=cb)

    def _on_exchange_declareok(self, frame: Method, userdata: str):
        self.log(logging.DEBUG, f'Exchange declared: {userdata}')
        self._start_publishing()

    def _start_publishing(self):
        self.log(logging.DEBUG, 'Issuing consumer related RPC commands')
        with self._lock:
            self._started = True

    def publish_message(self, message: dict | str | bytes, routing_key: str | None = None):
        if self._channel is None or not self._channel.is_open:
            return

        properties = pika.BasicProperties(app_id=self._owner_id,
                                          content_type='application/json')

        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if not isinstance(message, bytes):
            message = bytes(message, encoding='utf-8')

        self._channel.basic_publish(self._exchange,
                                    routing_key if routing_key is not None else self._routing_key,
                                    message,
                                    properties)
        self.log(logging.DEBUG, f'Published message {message}!')

    async def _wait_for_initialization(self):
        while True:
            with self._lock:
                if self._started:
                    break
            await asyncio.sleep(1.)

    async def start(self):
        self._setup_exchange()
        await self._wait_for_initialization()

    async def stop(self):
        if not self._stopping:
            self.log(logging.DEBUG, 'Stopping a publisher...')
            self._stopping = True
            self.log(logging.DEBUG, 'Publisher stopped!')


class RabbitMQPersistentConnection:
    def __init__(self,
                 owner_id: str,
                 host: str,
                 port: int,
                 log_func: Callable[[int, str], None],
                 prefetch_count: int = 1,
                 is_master: bool = False):
        self._reconnect_delay = 0

        self._owner_id = owner_id
        self._log_id = 'CORE'

        self._host = host
        self._port = port
        self.connection: AsyncioConnection | None = None
        self.channel: Channel | None = None
        self._prefetch_count = prefetch_count
        self._is_master = is_master

        self._consumers: list[RabbitMQAsyncioConsumer] = list()
        self._publishers: dict[tuple[str, str | None], RabbitMQAsyncioPublisher] = dict()

        self._log = log_func

        self._started = False
        self._lock = Lock()

    def log(self, level: int, message: str):
        return self._log(level, f'{self._log_id}: {message}')

    @property
    def is_closing_or_closed(self) -> bool:
        with self._lock:
            return self.connection.is_closed or self.connection.is_closing

    def _connect(self) -> AsyncioConnection:
        self.log(logging.DEBUG, f'Connecting to {self._host}:{self._port}...')
        time.sleep(self._reconnect_delay)
        if self._reconnect_delay < 30.:
            self._reconnect_delay += 1
        return AsyncioConnection(
            parameters=pika.ConnectionParameters(host=self._host, port=self._port),
            on_open_callback=self._on_connection_open,
            on_open_error_callback=self._on_connection_open_error,
            on_close_callback=self._on_connection_closed)

    def _on_connection_open(self, connection: AsyncioConnection):
        self.log(logging.DEBUG, 'Connection opened')
        self._open_channel()

    def _on_connection_open_error(self, connection: AsyncioConnection, err: Exception):
        self.log(logging.WARNING, f'Opening connection failed: {err}! Retrying...')
        self.connection = self._connect()

    def _on_connection_closed(self, connection: AsyncioConnection, reason: Exception):
        if isinstance(reason, ConnectionClosed) and reason.reply_code == 0:
            self.log(logging.DEBUG, f'Connection is closed by normal shutdown.')
        else:
            self.log(logging.WARNING, f'Connection unexpectedly closed, reconnecting. Reason: {reason}.')
            with self._lock:
                self._started = False
                self.connection = None
                self.channel = None
            self.connection = self._connect()

    def _open_channel(self):
        self.log(logging.DEBUG, 'Opening a new channel')
        self.connection.channel(on_open_callback=self._on_channel_open)

    def _on_channel_open(self, channel: Channel):
        self.log(logging.DEBUG, 'Channel opened')
        self.channel = channel
        self._add_on_channel_close_callback()
        with self._lock:
            self._reconnect_delay = 0.
            self._started = True

    def _add_on_channel_close_callback(self):
        self.log(logging.DEBUG, 'Adding channel close callback')
        self.channel.add_on_close_callback(self._on_channel_closed)

    def _on_channel_closed(self, channel: Channel, reason: Exception):
        if isinstance(reason, ChannelClosed) and reason.reply_code == 0:
            self.log(logging.DEBUG, f'Channel is closed by normal shutdown.')
            if not self.connection.is_closing and not self.connection.is_closed:
                self.connection.close(reply_code=0, reply_text='Normal shutdown')
        else:
            self.log(logging.WARNING, f'Channel {channel} unexpectedly closed, reconnecting. Reason {reason}.')
            with self._lock:
                self._started = False
                self.connection = None
                self.channel = None
            self.connection = self._connect()

    async def _wait_for_channel_open(self):
        while True:
            with self._lock:
                if self._started:
                    break
            await asyncio.sleep(1.)

    async def start(self):
        self.connection = self._connect()
        await self._wait_for_channel_open()

    async def add_consumer(self,
                           exchange_name: str,
                           exchange_type: ExchangeType,
                           routing_keys: str | Iterable[str],
                           msg_callback: Callable[[pika.channel.Channel, Basic.Deliver, BasicProperties, bytes], None],
                           prefetch_count: int | None = None):
        consumer = RabbitMQAsyncioConsumer(
            channel=self.channel,
            exchange_name=exchange_name,
            exchange_type=exchange_type,
            routing_keys=routing_keys,
            msg_callback=msg_callback,
            log_func=self.log,
            prefetch_count=prefetch_count if prefetch_count is not None else self._prefetch_count
        )
        await consumer.start()
        self._consumers.append(consumer)

    async def add_publisher(self,
                            exchange_name: str,
                            exchange_type: ExchangeType,
                            routing_key: str | None = None
                            ):
        publisher = RabbitMQAsyncioPublisher(
            owner_id=self._owner_id,
            channel=self.channel,
            exchange_name=exchange_name,
            exchange_type=exchange_type,
            routing_key=routing_key,
            log_func=self.log
        )
        await publisher.start()
        self._publishers[(exchange_name, routing_key)] = publisher

    def publish_message(self,
                        exchange_name: str,
                        routing_key: str | None,
                        message: dict | str | bytes):
        publisher_key = (exchange_name, None) if (exchange_name, None) in self._publishers else (exchange_name, routing_key)
        self._publishers[publisher_key].publish_message(message, routing_key)

    async def stop(self):
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
