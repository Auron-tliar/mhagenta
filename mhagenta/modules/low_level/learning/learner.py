from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Observation, State, Belief
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

    def send_model(self, reasoner_id: str, model: Any, **kwargs) -> None:
        """Send a learned model to a low-level reasoner.

        Args:
            reasoner_id (str): `module_id` of the relevant low-level reasoner.
            model (Any): model to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'model': model}
        if kwargs:
            body.update(kwargs)
        self._add(reasoner_id, ConnType.send, body)


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
            sender (str): `module_id` of the reasoner that sent the learning task.
            task (Any): received learning task object.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            LearnerState: modified or unaltered internal state of the module.

        """
        return state

    def on_memories(self, state: LearnerState, sender: str, memories: Iterable[Belief | Observation], **kwargs) -> LearnerState:
        """Override to define learner's reaction to receiving a collection of memories.

        Args:
            state (LearnerState): Learner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the memory structure that send the memories.
            memories (Iterable[Belief | Observation]): received collection of memories.
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
            sender (str): `module_id` of the reasoner that sent the model request.
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

        for ll_reasoner in self._directory.internal.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner.module_id, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner.module_id, ConnType.request, self._receive_model_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner.module_id, ConnType.send, self._receive_task))

        for hl_reasoner in self._directory.internal.hl_reasoning:
            out_id_channels.append(self.sender_reg_entry(hl_reasoner.module_id, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner.module_id, ConnType.request, self._receive_model_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner.module_id, ConnType.send, self._receive_task))

        for memory in self._directory.internal.memory:
            out_id_channels.append(self.sender_reg_entry(memory.module_id, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(memory.module_id, ConnType.send, self._receive_memories))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=LearnerOutbox
        )

    def _receive_task(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.debug(f'Received a new task {msg.id} from {sender}. Processing...')
        task = msg.body.pop('task')
        update = self._base.on_task(state=self._state, sender=sender, task=task, **msg.body)
        self.log(5, f'Finished processing the new task {msg.id}!')
        return update

    def _receive_memories(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.debug(f'Received memories {msg.id} from {sender}. Processing...')
        memories = msg.body.pop('memories')
        update = self._base.on_memories(state=self._state, sender=sender, memories=memories, **msg.body)
        self.log(5, f'Finished processing memories {msg.id}!')
        return update

    def _receive_model_request(self, sender: str, channel: str, msg: Message) -> LearnerState:
        self.debug(f'Received a model request {msg.id} from {sender}. Processing...')
        update = self._base.on_model_request(state=self._state, sender=sender, **msg.body)
        self.log(5, f'Finished processing the model request {msg.id}!')
        return update
