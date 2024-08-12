from typing import ClassVar, Iterable

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Belief, State
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class KnowledgeOutbox(Outbox):
    def send_memories(self, memory_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(memory_id, ConnType.send, body)

    def send_beliefs(self, receiver: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(receiver, ConnType.send, body)


class KnowledgeBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.KNOWLEDGE

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        raise NotImplementedError()

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
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
            module_id=self._base.module_id,
            module_type=self._base.module_type,
            initial_state=self._base.initial_state,
            step_action=self._base.step if not self._base.is_reactive else None,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks
        )

    def _receive_beliefs(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.debug(f'Finished processing belief update {msg.id}!')
        return update

    def _receive_belief_request(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received beliefs request {msg}. Processing...')
        update = self._base.on_belief_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing beliefs request {msg.id}!')
        return update