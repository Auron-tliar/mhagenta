from typing import ClassVar, Iterable

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Belief, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class KnowledgeOutbox(Outbox):
    """Internal communication outbox class for Knowledge model.

    Used to store and process outgoing messages to other modules.

    """
    def send_memories(self, memory_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        """Send an outdated belief collection as a memory to a memory structure.

        Args:
            memory_id (str): `module_id` of the relevant memory structure.
            beliefs (Iterable[Belief]): a collection of beliefs to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(memory_id, ConnType.send, body)

    def send_beliefs(self, hl_reasoner_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        """Send a collection of beliefs to a high-level reasoner.

        Args:
            hl_reasoner_id (str): `module_id` of the relevant high-level reasoner structure.
            beliefs: a collection of beliefs to send.
            **kwargs: additional keyword arguments to be included in the message.

        """

        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(hl_reasoner_id, ConnType.send, body)


KnowledgeState = State[KnowledgeOutbox]


class KnowledgeBase(ModuleBase):
    """Base class for defining Knowledge model behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.KNOWLEDGE

    def on_belief_update(self, state: KnowledgeState, sender: str, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        """Override to define knowledge model's reaction to receiving a belief update.

        Args:
            state (KnowledgeState): Knowledge model's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the module (low-level or high-level reasoner) that sent the beliefs.
            beliefs (Iterable[Belief]): received collection of beliefs.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            KnowledgeState: modified or unaltered internal state of the module.

        """
        return state

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs) -> KnowledgeState:
        """Override to define knowledge model's reaction to receiving a belief request.

        Args:
            state (KnowledgeState): Knowledge model's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the high-level reasoner that sent the request.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            KnowledgeState: modified or unaltered internal state of the module.

        """
        return state


class Knowledge(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: KnowledgeBase
                 ) -> None:
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.send, self._receive_beliefs))

        for memory in self._directory.memory:
            out_id_channels.append(self.sender_reg_entry(memory, ConnType.send))

        for hl_reasoner in self._directory.hl_reasoning:
            out_id_channels.append(self.sender_reg_entry(hl_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner, ConnType.request, self._receive_belief_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner, ConnType.send, self._receive_beliefs))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=KnowledgeOutbox
        )

    def _receive_beliefs(self, sender: str, channel: str, msg: Message) -> KnowledgeState:
        self.info(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.debug(f'Finished processing belief update {msg.id}!')
        return update

    def _receive_belief_request(self, sender: str, channel: str, msg: Message) -> KnowledgeState:
        self.info(f'Received beliefs request {msg}. Processing...')
        update = self._base.on_belief_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing beliefs request {msg.id}!')
        return update
