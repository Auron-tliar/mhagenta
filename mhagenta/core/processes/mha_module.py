import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal

import dill
from pydantic import BaseModel, ConfigDict

from mhagenta.core.processes.process import MHAProcess
from mhagenta.utils import AgentCmd, StatusReport, Message, Outbox, State, Directory
from mhagenta.utils.common import DEFAULT_LOG_FORMAT
from mhagenta.utils.common.typing import MessageCallback, MsgProcessorCallback, Sender, Channel, Recipient
from mhagenta.core.connection import ModuleMessenger, Connector


class GlobalParams(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    agent_id: str
    directory: Directory
    connector_cls: type[Connector]
    connector_kwargs: dict[str, Any] = dict()
    step_frequency: float = .1
    status_frequency: float = 5.
    control_frequency: float = .05
    agent_start_time: float
    exec_duration: float = 60.
    save_dir: str
    save_format: Literal['json', 'dill'] = 'json'
    resume: bool = False
    log_level: int | str = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT


class ModuleBase:
    """Base class for agent module definitions. Contains templates for all basic functions that define module behaviour.

    """
    module_type: ClassVar[str]

    def __init__(self,
                 module_id: str,
                 initial_state: dict[str, Any] | None = None,
                 init_kwargs: dict[str, Any] | None = None
                 ) -> None:
        """ModuleBase constructor.

        Args:
            module_id (str): unique (in scope of an agent) ID of the module.
            initial_state (dict[str, Any], optional): dictionary of fields and corresponding values to be inserted into
                module's internal state at initialization. Later on can be accessed via State.field.
            init_kwargs (dict[str, Any], optional): keyword arguments to be passed to the `on_init` method.
        """
        self.module_id = module_id
        self.initial_state = initial_state
        self.init_kwargs = init_kwargs if init_kwargs is not None else dict()
        self._is_reactive = self._check_reactive()

    def step(self, state: State) -> State:
        """Base for module's step function. If not overridden, the module will NOT take periodic step actions.

        Args:
            state (State): module's internal state enriched with relevant runtime information and functionality.

        Returns:
            State: modified or unaltered internal state of the module.

        """
        return state

    def on_init(self, **kwargs) -> None:
        """Called after the module finished initializing

        Args:
            **kwargs: additional keyword arguments. Their values need to be passed to the module constructor.

        """
        pass

    def on_first(self, state: State) -> State:
        """Called right after agent's execution start, before the first call to the regular step function.

        Args:
            state: module's internal state enriched with relevant runtime information and functionality.

        Returns:
            State: modified or unaltered internal state of the module.

        """
        return state

    def on_last(self, state: State) -> State:
        """Called right after agent's execution stop is initiated, the last behavioral action module can take.

        Args:
            state: module's internal state enriched with relevant runtime information and functionality.

        Returns:
            State: modified or unaltered internal state of the module.

        """
        return state

    def _check_reactive(self) -> bool:
        source_class = getattr(self, 'step').__qualname__.partition('.')[0]
        return source_class == ModuleBase.__name__

    @property
    def is_reactive(self) -> bool:
        """Shows whether the module is reactive (i.e. if it has internal action loop or just reacts to communications).

        Returns:
            bool: True if the module is reactive, otherwise False.

        """
        return self._is_reactive


class MHAModule(MHAProcess):
    def __init__(self,
                 global_params: GlobalParams,
                 base: ModuleBase,
                 out_id_channels: Iterable[tuple[Recipient, Channel]],
                 in_id_channel_callbacks: Iterable[tuple[Sender, Channel, MessageCallback]],
                 outbox_cls: type[Outbox],
                 ) -> None:
        super().__init__(
            agent_id=global_params.agent_id,
            agent_start_time=global_params.agent_start_time,
            exec_start_time=None,
            exec_duration=global_params.exec_duration,
            step_frequency=global_params.step_frequency,
            control_frequency=global_params.control_frequency,
            log_id=base.module_id,
            log_level=global_params.log_level,
            log_format=global_params.log_format
        )

        self._base = base

        self._module_id = self._base.module_id

        if self._base.initial_state is None:
            self._base.initial_state = dict()
        self._state = State[outbox_cls](
            agent_id=global_params.agent_id,
            module_id=self._base.module_id,
            time_func=self._time.get_exec_time,
            directory=global_params.directory,
            outbox=outbox_cls(),
            **self._base.initial_state)
        self._status_frequency = global_params.status_frequency

        self._save_dir = global_params.save_dir
        self._save_format = global_params.save_format
        if global_params.resume:
            self.load_state()

        self._step_action = self._base.step if not self._base.is_reactive else None
        self._step_counter = 0

        self._messenger = ModuleMessenger(
            connector_cls=global_params.connector_cls,
            agent_id=global_params.agent_id,
            module_type=self._base.module_type,
            module_id=self._module_id,
            agent_time=self._time,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=[(sender, channel, self._on_msg_task_generator(callback))
                                     for sender, channel, callback in in_id_channel_callbacks],
            agent_cmd_callback=self._on_cmd_task,
            log_tags=self._log_tags,
            log_level=global_params.log_level,
            log_format=global_params.log_format,
            **global_params.connector_kwargs
        )

    async def on_init(self) -> None:
        await self._messenger.initialize()
        self._base.on_init(**self._base.init_kwargs)

    async def on_start(self) -> None:
        self._task_group.create_task(self._messenger.start())
        self._queue.push(
            func=self._report_status,
            ts=self._time.agent,
            priority=True,
            periodic=True,
            frequency=self._status_frequency
        )

    def _on_step_task(self) -> None:
        try:
            self.debug(f'Running step {self._step_counter} [[[{self._state}]]]...')
            self._step_counter += 1
            update = self._step_action(self._state)
            self._process_update(update)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while running the step action {self._step_counter}!'
                         f' Aborting step action {self._step_counter} and attempting to resume execution...')
            raise ex

    def _run(self) -> None:
        self._stage = self.Stage.running

    async def on_run(self) -> None:
        self._on_first_step()
        if not self._base.is_reactive:
            self._queue.push(
                func=self._on_step_task,
                ts=self._time.agent,
                periodic=True,
                frequency=self._step_frequency
            )

    def _on_first_step(self) -> None:
        try:
            self.debug(f'Running first (pre) step...')
            self._step_counter += 1
            update = self._base.on_first(self._state)
            self._process_update(update)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while running the first (pre) step action!'
                         f' Aborting and attempting to resume execution...')
            raise ex

    def _on_last_step(self) -> None:
        try:
            self.debug(f'Running last (post) step...')
            self._step_counter += 1
            update = self._base.on_last(self._state)
            self._process_update(update)
        except Exception as ex:
            self.warning(f'Caught exception \"{ex}\" while running the last (post) step action!'
                         f' Aborting and attempting to resume execution...')
            raise ex

    async def on_stop(self) -> None:
        self.info('Stopping')
        self._on_last_step()
        self.save_state()
        self._queue.clear()
        self._report_status()
        await self._messenger.stop()

    def on_cmd(self, cmd: AgentCmd) -> None:
        if self._agent_id != cmd.agent_id:
            return
        match cmd.cmd:
            case cmd.START:
                self.info(f'Received {cmd.START} command (start ts: {cmd.args["start_ts"] if "start_ts" in cmd.args else "-"})')
                self._time.set_exec_start_ts(cmd.args['start_ts'] if 'start_ts' in cmd.args else self._time.agent)
                self._stop_time = cmd.args['start_ts'] + self._exec_duration
                # self._stage = self.Stage.starting
                self._queue.push(
                    func=self._run,
                    ts=self._time.exec_start_ts,
                    priority=True,
                    periodic=False
                )
            case cmd.STOP:
                self.info(f'Received {cmd.STOP} command (reason: {cmd.args["reason"]})')
                self._stage = self.Stage.stopping
                self._stop_reason = cmd.args['reason']
            case _:
                self.warning(f'Received unknown command {cmd.cmd}! Ignoring...')

    def _on_cmd_task(self, task_cmd: AgentCmd) -> None:
        self._queue.push(
            func=self.on_cmd,
            ts=self._time.agent,
            priority=True,
            cmd=task_cmd
        )

    def _on_msg_task_generator(self, callback: MessageCallback) -> MsgProcessorCallback:
        def push_task(task_sender: str, task_channel: str, task_msg: Message) -> None:
            def on_msg_task(sender: str, channel: str, msg: Message) -> None:
                try:
                    update = callback(sender, channel, msg)
                    self._process_update(update)
                except Exception as ex:
                    self.warning(f'Caught exception \"{ex}\" while processing message {msg.short_id} from {sender} (channel: {channel})!'
                                 f' Aborting message processing and attempting to resume execution...')
                    raise ex
            self._queue.push(
                func=on_msg_task,
                ts=task_msg.ts,
                sender=task_sender,
                channel=task_channel,
                msg=task_msg
            )

        return push_task

    def _process_update(self, update: State) -> None:
        self._state = update
        if self._state.outbox:
            self._process_outbox()

    def _process_outbox(self) -> None:
        for receiver, performative, extension, content in self._state.outbox:
            self.debug(f'SENDING {performative.capitalize()}{f"/{extension}" if extension else ""} TO {receiver}: {content}...')
            self._send(receiver, performative, extension, content)
        self._state.outbox.clear()

    def _send(self, recipient: str, performative: str, extension: str, content: dict[str, Any]) -> None:
        channel = self.sender_channel(recipient, performative, extension)
        msg = Message(
            body=content,
            sender_id=self._module_id,
            recipient_id=recipient,
            ts=self._time.agent,
            performative=performative
        )
        self._messenger.send(recipient, channel, msg)

    @property
    def status(self) -> StatusReport:
        args = dict()
        if self._error_status is not None:
            status_str = StatusReport.ERROR
            args['error'] = self._format_exception(self._error_status)
            self._error_status = None
        elif self._stage < self.Stage.ready:
            status_str = StatusReport.CREATED
        elif self._stage < self.Stage.running:
            status_str = StatusReport.READY
        elif self._stage < self.Stage.stopping:
            status_str = StatusReport.RUNNING
        else:
            status_str = StatusReport.FINISHED
        return StatusReport(
            agent_id=self._agent_id,
            module_id=self._module_id,
            status=status_str,
            ts=self._time.agent,
            args=args
        )

    def _report_status(self) -> None:
        self.debug(f'Reporting {self.status}...')
        self._messenger.report_status(self.status)

    async def on_error(self, error: Exception) -> None:
        await super().on_error(error)
        self._report_status()

    # def await_start_cond(self) -> bool:
    #     if self._stage >= self.Stage.running:
    #         return True
    #
    #     if self._time.exec is None:
    #         return self._time.agent >= self._stop_time
    #     else:
    #         return self._time.exec >= 0
    #
    # async def await_exec_start(self) -> None:
    #     if self._stage >= self.Stage.running:
    #         return
    #
    #     if self._time.exec is not None and self._time.exec >= 0:
    #         self._stage = self.Stage.running
    #         if self._step_action is not None:
    #             self._queue.push(
    #                 func=self._on_step_task,
    #                 ts=self._time.agent,
    #                 periodic=True,
    #                 frequency=self._step_frequency
    #             )

    @staticmethod
    def channel_name(sender: str, recipient: str, conn_type: str, extension: str = '') -> str:
        return f'{sender}_{recipient}_{conn_type}{f"_{extension}" if extension else ""}'

    def sender_channel(self, recipient: str, conn_type: str, extension: str = '') -> str:
        return f'{self._module_id}_{recipient}_{conn_type}{f"_{extension}" if extension else ""}'

    def sender_reg_entry(self, receiver: str, conn_type: str, extension: str = '') -> tuple[str, str]:
        return receiver, self.sender_channel(receiver, conn_type, extension)

    def recipient_channel(self, sender: str, conn_type: str, extension: str = '') -> str:
        return f'{sender}_{self._module_id}_{conn_type}{f"_{extension}" if extension else ""}'

    def recipient_reg_entry(self, sender: str, conn_type: str, callback: MessageCallback, extension: str = '') -> tuple[Sender, Channel, MessageCallback]:
        return sender, self.recipient_channel(sender, conn_type, extension), callback

    def save_state(self) -> None:
        path = Path(self._save_dir)
        path.mkdir(exist_ok=True)
        path /= f'{self._agent_id}.{self._module_id}.sav'
        match self._save_format:
            case 'json':
                path = path.with_suffix('.json')
                with open(path, 'w') as f:
                    json.dump(self._state.dump(), f)
            case 'dill':
                with open(path, 'wb') as f:
                    dill.dump(self._state.dump(), f)
            case _:
                raise ValueError(f'Unsupported save format: {self._save_format}!')

    def load_state(self) -> None:
        path = Path(self._save_dir) / f'{self._agent_id}.{self._module_id}.sav)'
        match self._save_format:
            case 'json':
                path = path.with_suffix('.json')
                with open(path, 'r') as f:
                    state = json.load(f)
            case 'dill':
                with open(path, 'rb') as f:
                    state = dill.load(f)
            case _:
                raise ValueError(f'Unsupported save format: {self._save_format}!')

        self._state.load(**state)


async def run_agent_module(module: type[MHAModule], *args, **kwargs) -> str | int:
    agent_module = module(*args, **kwargs)
    await agent_module.initialize()
    exit_reason = await agent_module.start()

    return exit_reason
