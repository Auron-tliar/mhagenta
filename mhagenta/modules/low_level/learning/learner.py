from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Observation, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class LearnerOutbox(Outbox):
    """Internal communication outbox class for Learner.

    Used to store and process outgoing messages to other modules.

    """
    def request_memories(self, memory_id: str, **kwargs) -> None:
        """Request a collection of memories from a memory structure.

        Args:
            memory_id (str): `module_id` of the relevant memory structure.
            **kwargs: additional keyword arguments to be included in the message.

        """
        self._add(memory_id, ConnType.request, kwargs)

    def send_status(self, ll_reasoner_id: str, learning_status: Any, **kwargs) -> None:
        """Send learning status to a low-level reasoner.

        Args:
            ll_reasoner_id (str): `module_id` of the low-level reasoner to report to.
            learning_status (Any): learning status to report.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'learning_status': learning_status}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body, extension='status')

    def send_model(self, ll_reasoner_id: str, model: Any, **kwargs) -> None:
        """Send a learned model to a low-level reasoner.

        Args:
            ll_reasoner_id (str): `module_id` of the relevant low-level reasoner.
            model (Any): model to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'model': model}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body, extension='model')


LearnerState = State[LearnerOutbox]


class LearnerBase(ModuleBase):
    """Base class for defining Learner behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.LEARNER

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs) -> LearnerState:
        """Override to define learner's reaction to receiving a learning task.

        Args:
            state (LearnerState): Learner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the low-level reasoner that sent the learning task.
            task (Any): received learning task object.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            LearnerState: modified or unaltered internal state of the module.

        """
        return state

    def on_memories(self, state: LearnerState, sender: str, observations: Iterable[Observation], **kwargs) -> LearnerState:
        """Override to define learner's reaction to receiving a collection of memories.

        Args:
            state (LearnerState): Learner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the memory structure that send the memories.
            observations (Iterable[Observation]): received collection of memories (observation objects).
            **kwargs: additional keyword arguments included in the message.

        Returns:
            LearnerState: modified or unaltered internal state of the module.

        """
        return state

    def on_model_request(self, state: LearnerState, sender: str, **kwargs) -> LearnerState:
        """Override to define learner's reaction to receiving a model request.

        Args:
            state (LearnerState): Learner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the low-level reasoner that sent the model request.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            LearnerState: modified or unaltered internal state of the module.

        """
        return state


class Learner(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: LearnerBase
                 ):
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send, extension='model'))
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send, extension='status'))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.request, self._receive_model_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.send, self._receive_task))

        for memory in self._directory.memory:
            out_id_channels.append(self.sender_reg_entry(memory, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(memory, ConnType.send, self._receive_memories))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=LearnerOutbox
        )

    def _receive_task(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.info(f'Received a new task {msg.id} from {sender}. Processing...')
        task = msg.body.pop('task')
        update = self._base.on_task(state=self._state, sender=sender, task=task, **msg.body)
        self.debug(f'Finished processing the new task {msg.id}!')
        return update

    def _receive_memories(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.info(f'Received memories {msg.id} from {sender}. Processing...')
        observations = msg.body.pop('observations')
        update = self._base.on_memories(state=self._state, sender=sender, observations=observations, **msg.body)
        self.debug(f'Finished processing memories {msg.id}!')
        return update

    def _receive_model_request(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.info(f'Received a model request {msg.id} from {sender}. Processing...')
        update = self._base.on_model_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing the model request {msg.id}!')
        return update
