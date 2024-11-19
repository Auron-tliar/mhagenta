import asyncio
import copy
import heapq
import logging
import time
import traceback
from abc import ABC
from enum import Enum
from functools import total_ordering
from types import NoneType
from typing import Callable, Any, Self

from mhagenta.utils.common.classes import MHABase, AgentTime
from mhagenta.utils.common import DEFAULT_LOG_FORMAT, LoggerExtras


@total_ordering
class Task:
    def __init__(self,
                 func: Callable,
                 ts: float,
                 args_getter: NoneType | Callable[[], tuple[list, dict[str, Any]]] = None,
                 priority: bool = False,
                 periodic: bool = False,
                 frequency: float = 0.,
                 stop_condition: Callable[[], bool] = None,
                 *args, **kwargs) -> None:
        self._func = func
        self._ts = ts
        self._args_getter = args_getter
        self._priority = priority
        self._periodic = periodic
        self._frequency = frequency
        self._stop_condition = stop_condition
        self._args = args
        self._kwargs = kwargs

    @property
    def ts(self) -> float:
        return self._ts

    @ts.setter
    def ts(self, ts: float) -> None:
        self._ts = ts

    @property
    def priority(self) -> bool:
        return self._priority

    @property
    def periodic(self) -> bool:
        return self._periodic

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def stop_check(self) -> bool:
        if self._stop_condition is None:
            return False
        return self._stop_condition()

    def __call__(self) -> Any:
        if self._args_getter is None:
            return self._func(*self._args, **self._kwargs)
        else:
            args, kwargs = self._args_getter()
            return self._func(*args, **kwargs)

    def __eq__(self, other) -> bool:
        return self.ts == other.ts

    def __gt__(self, other) -> bool:
        return self.ts > other.ts

    @classmethod
    def cmp_func(cls) -> Callable[[Self], float]:
        return lambda task: task.cmp_key

    @property
    def queue_item(self) -> tuple[float, Self]:
        return self.ts, self

    def copy(self, ts: float) -> Self:
        task = copy.copy(self)
        task.ts = ts
        return task

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(\'{self._func.__name__}\', {self._ts}{", PRIORITY" if self.priority else ""}{f", PERIODIC({self.frequency})" if self.periodic else ""})'

    def __repr__(self) -> str:
        return str(self)


class ExecQueue:
    def __init__(self, agent_start_ts: float) -> None:
        self._start_ts = agent_start_ts
        self._queue: list[tuple[float, Task]] = list()
        self._priority_queue: list[tuple[float, Task]] = list()

    def push(self,
             func: Callable,
             ts: float,
             priority: bool = False,
             args_getter: NoneType | Callable[[], tuple[list, dict[str, Any]]] = None,
             periodic: bool = False,
             frequency: float = 0.,
             stop_condition: Callable[[], bool] = None,
             *args,
             **kwargs
             ) -> None:
        task = Task(
            func=func,
            ts=ts,
            args_getter=args_getter,
            priority=priority,
            periodic=periodic,
            frequency=frequency,
            stop_condition=stop_condition,
            *args,
            **kwargs
        )
        if priority:
            heapq.heappush(self._priority_queue, task.queue_item)
        else:
            heapq.heappush(self._queue, task.queue_item)

    def peek(self) -> Task | None:
        if self._priority_queue and self._priority_queue[0][0] < self._agent_time:
            return self._priority_queue[0][1]
        if self._queue and self._queue[0][0] < self._agent_time:
            return self._priority_queue[0][1]
        if self._priority_queue:
            if self._queue:
                if self._priority_queue[0][0] <= self._queue[0][0]:
                    return self._priority_queue[0][1]
                else:
                    return self._queue[0][1]
            else:
                return self._priority_queue[0][1]
        elif self._queue:
            return self._queue[0][1]
        else:
            return None

    @property
    def pending(self) -> bool:
        return ((self._priority_queue and self._priority_queue[0][0] < self._agent_time) or
                (self._queue and self._queue[0][0] < self._agent_time))

    # def pop(self, priority: bool = False, stop: bool = False) -> Task | None:
    #     if self._priority_queue and self._priority_queue[0][0] < self._agent_time:
    #         task: Task = heapq.heappop(self._priority_queue)[1]
    #     elif not priority and self._queue and self._queue[0][0] < self._agent_time:
    #         task: Task = heapq.heappop(self._queue)[1]
    #     else:
    #         return None
    #
    #     if task.periodic and not stop and not task.stop_check:
    #         if task.priority:
    #             heapq.heappush(self._priority_queue, self._periodic_add_task(task).queue_item)
    #         else:
    #             heapq.heappush(self._queue, self._periodic_add_task(task).queue_item)
    #         # next_task = task.copy(task.ts + task.frequency)
    #         # if task.priority:
    #         #     heapq.heappush(self._priority_queue, next_task.queue_item)
    #         # else:
    #         #     heapq.heappush(self._queue, next_task.queue_item)
    #
    #     return task

    def run_next(self, priority: bool = False, stop: bool = False) -> None:
        if self._priority_queue and self._priority_queue[0][0] < self._agent_time:
            task: Task = heapq.heappop(self._priority_queue)[1]
        elif not priority and self._queue and self._queue[0][0] < self._agent_time:
            task: Task = heapq.heappop(self._queue)[1]
        else:
            return None

        task()

        if task.periodic and not stop and not task.stop_check:
            next_task = task.copy(max(task.ts + task.frequency, self._agent_time))
            if task.priority:
                heapq.heappush(self._priority_queue, next_task.queue_item)
            else:
                heapq.heappush(self._queue, next_task.queue_item)

    def __len__(self) -> int:
        return len(self._queue) + len(self._priority_queue)

    def __bool__(self) -> bool:
        return (bool(self._priority_queue) and self._priority_queue[0][0] < self._agent_time) or (bool(self._queue) and self._queue[0][0] < self._agent_time)

    def next_wait(self) -> float:
        if self._priority_queue:
            priority_wait = self._priority_queue[0][0] - self._agent_time
        else:
            priority_wait = float('inf')

        if self._queue:
            regular_wait = self._queue[0][0] - self._agent_time
        else:
            regular_wait = float('inf')

        return max(0., min(priority_wait, regular_wait))

    @property
    def scheduled(self) -> bool:
        return bool(self._queue) or bool(self._priority_queue)

    @property
    def empty(self) -> bool:
        return not self.pending

    def clear(self, priority: bool = True) -> None:
        self._queue = list()
        if priority:
            self._priority_queue = list()

    @property
    def _agent_time(self) -> float:
        return time.time() - self._start_ts

    def _periodic_add_task(self, task: Task) -> Task:
        def add() -> None:
            if task.stop_check:
                return
            next_task = task.copy(max(task.ts + task.frequency, self._agent_time))
            if task.priority:
                heapq.heappush(self._priority_queue, next_task.queue_item)
            else:
                heapq.heappush(self._queue, next_task.queue_item)
        return Task(
            func=add,
            ts=task.ts,
            priority=task.priority,
            periodic=False
        )

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(Priority: {self._priority_queue}, regular; {self._queue})'

    def __repr__(self) -> str:
        return str(self)


class MHAProcess(MHABase, ABC):
    @total_ordering
    class Stage(Enum):
        created = 10
        initializing = 20
        initialized = 40
        starting = 60
        ready = 80
        running = 100
        stopping = 120
        stopped = 140

        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                raise NotImplemented('Stage object can only be compared with another Status object!')
            return self.value == other.value

        def __lt__(self, other):
            if not isinstance(other, self.__class__):
                raise NotImplemented('Stage object can only be compared with another Status object!')
            return self.value < other.value

    def __init__(self,
                 agent_id: str,
                 agent_start_time: float,
                 exec_start_time: float | None = None,
                 init_timeout: float = 60.,
                 exec_duration: float = 60.,
                 step_frequency: float = 0.,
                 control_frequency: float = .5,
                 log_id: str | None = None,
                 log_level: int | str = logging.DEBUG,
                 log_format: str = DEFAULT_LOG_FORMAT
                 ) -> None:
        assert step_frequency >= 0
        assert agent_start_time > 0
        assert exec_start_time is None or exec_start_time > 0

        super().__init__(
            agent_id=agent_id,
            log_id=log_id,
            log_tags=None,
            log_level=log_level,
            log_format=log_format
        )

        self._exec_duration = exec_duration
        if exec_start_time is None:
            start_time = None
            self._stop_time = init_timeout
        else:
            start_time = exec_start_time
            self._stop_time = exec_start_time + exec_duration
        self._step_frequency = step_frequency
        self._control_frequency = control_frequency

        self._time = AgentTime(
            agent_start_ts=agent_start_time,
            exec_start_ts=start_time,
        )

        self._queue = ExecQueue(agent_start_ts=self._time.agent_start_ts)
        self._loop_counter = 0

        self._task_group: asyncio.TaskGroup | None = None
        self._main_loop: asyncio.Task | None = None

        self._stage = self.Stage.created
        self._error_status: None | Exception = None

        self._stop_reason: str = ''

    async def initialize(self) -> None:
        if self._stage >= self.Stage.initializing:
            return
        self._stage = self.Stage.initializing
        await self.on_init()
        self._stage = self.Stage.initialized

    async def start(self) -> str:
        if self._stage < self.Stage.initializing:
            await self.initialize()

        self._stage = self.Stage.starting

        async with asyncio.TaskGroup() as tg:
            self._task_group = tg
            self._main_loop = tg.create_task(self._loop())
            tg.create_task(self.on_start())
            self._stage = self.Stage.ready
        await self.stop(self._stop_reason)
        return self._stop_reason

    async def stop(self, reason: str = 'USER COMMAND') -> None:
        self._stop_reason = reason
        self.info(f'Stopping! Reason: {reason}.')
        self._stage = self.Stage.stopping
        await self.on_stop()
        self._main_loop.cancel()
        self._stage = self.Stage.stopped
        self.info('Stopped!')

    async def _loop(self) -> None:
        prev_stage = self._stage
        self._loop_counter = -1
        while self._time.agent < self._stop_time:
            try:
                self._loop_counter += 1

                if self._stage != prev_stage:
                    await self.on_stage_change()
                prev_stage = self._stage

                # if  self._queue.pending:
                #     self.log(5, f'Executing task: {self._queue.peek()}. Queue length: {len(self._queue)}.')
                self._queue.run_next(priority=(self._stage != self.Stage.running))
                # if task is not None:
                #     task()

                # self._error_status = None
            except Exception as ex:
                await self.on_error(ex)
            await asyncio.sleep(min(self._control_frequency, self._queue.next_wait()))
        self._stop_reason = 'TIMEOUT'

    async def on_stage_change(self) -> None:
        match self._stage:
            case self.Stage.starting:
                await self.on_start()
            case self.Stage.running:
                await self.on_run()
            case self.Stage.stopping:
                await self.stop()

    async def on_init(self) -> None:
        pass

    async def on_start(self) -> None:
        pass

    async def on_run(self) -> None:
        pass

    async def on_stop(self) -> None:
        pass

    async def on_error(self, error: Exception) -> None:
        self._error_status = error

    @property
    def time(self) -> AgentTime:
        return self._time

    @property
    def _logger_extras(self) -> LoggerExtras | None:
        return LoggerExtras(
            agent_time=self._time.agent,
            mod_time=self._time.module,
            exec_time=str(self._time.exec) if self._time.exec is not None else '-',
            tags=self.log_tag_str
        )

    @staticmethod
    def _format_exception(error: Exception) -> str:
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
