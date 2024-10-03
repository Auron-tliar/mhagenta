from typing import ClassVar, Iterable

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Belief, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class KnowledgeOutbox(Outbox):
    def send_memories(self, memory_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(memory_id, ConnType.send, body)

    def send_beliefs(self, knowledge_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(knowledge_id, ConnType.send, body)


KnowledgeState = State[KnowledgeOutbox]


class KnowledgeBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.KNOWLEDGE

    def on_belief_update(self, state: KnowledgeState, sender: str, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        raise NotImplementedError()

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs) -> KnowledgeState:
        raise NotImplementedError()


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
