from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Observation, Belief, State
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class MemoryOutbox(Outbox):
    def send_observations(self, learner_id: str, observations: Iterable[Any], **kwargs) -> None:
        body = {'observations': observations}
        if kwargs:
            body.update(kwargs)
        self._add(learner_id, ConnType.send, body)

    def send_beliefs(self, hl_reasoner_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(hl_reasoner_id, ConnType.send, body)


class MemoryBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.MEMORY

    def on_observation_request(self, state: State, sender: str, **kwargs) -> Update:
        raise NotImplementedError()

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
        raise NotImplementedError()

    def on_observation_update(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        raise NotImplementedError()

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        raise NotImplementedError()


class Memory(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: MemoryBase
                 ):
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.send, self._receive_observations))

        for learner in self._directory.learning:
            out_id_channels.append(self.sender_reg_entry(learner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(learner, ConnType.request, self._receive_observation_request))

        for knowledge in self._directory.knowledge:
            in_id_channels_callbacks.append(self.recipient_reg_entry(knowledge, ConnType.send, self._receive_beliefs))

        for hl_reasoner in self._directory.hl_reasoning:
            out_id_channels.append(self.sender_reg_entry(hl_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner, ConnType.request, self._receive_belief_request))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks
        )

    def _receive_observations(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received observation update {msg.id} from {sender}. Processing...')
        observations = msg.body.pop('observations')
        update = self._base.on_observation_update(state=self._state, sender=sender, observations=observations, **msg.body)
        self.debug(f'Finished processing observation update {msg.id}!')
        return update

    def _receive_beliefs(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.debug(f'Finished processing belief update {msg.id}!')
        return update

    def _receive_observation_request(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received observation request {msg.id} from {sender}. Processing...')
        update = self._base.on_observation_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing observation request {msg.id}!')
        return update

    def _receive_belief_request(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received belief request {msg.id} from {sender}. Processing...')
        update = self._base.on_belief_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing belief request {msg.id}!')
        return update
