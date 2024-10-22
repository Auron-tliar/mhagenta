from typing import Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Goal, Belief, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class HLOutbox(Outbox):
    def request_beliefs(self, knowledge_id: str, **kwargs) -> None:
        self._add(knowledge_id, ConnType.request, kwargs)

    def request_memories(self, memory_id: str, **kwargs) -> None:
        self._add(memory_id, ConnType.request, kwargs)

    def request_action(self, actuator_id: str, **kwargs) -> None:
        self._add(actuator_id, ConnType.request, kwargs)

    def send_beliefs(self, knowledge_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(knowledge_id, ConnType.send, body)

    def send_goals(self, goal_graph_id: str, goals: Iterable[Goal], **kwargs) -> None:
        body = {'goals': goals}
        if kwargs:
            body.update(kwargs)
        self._add(goal_graph_id, ConnType.send, body)


HLState = State[HLOutbox]


class HLReasonerBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.HLREASONER

    def on_belief_update(self, state: HLState, sender: str, beliefs: Iterable[Belief], **kwargs) -> HLState:
        raise NotImplementedError()

    def on_goal_update(self, state: HLState, sender: str, goals: Iterable[Goal], **kwargs) -> HLState:
        raise NotImplementedError()


class HLReasoner(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: HLReasonerBase
                 ):
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for knowledge in self._directory.knowledge:
            out_id_channels.append(self.sender_reg_entry(knowledge, ConnType.request))
            out_id_channels.append(self.sender_reg_entry(knowledge, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(knowledge, ConnType.send, self._receive_belief_update))

        for memory in self._directory.memory:
            out_id_channels.append(self.sender_reg_entry(memory, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(memory, ConnType.send, self._receive_belief_update))

        for goal_graph in self._directory.goals:
            out_id_channels.append(self.sender_reg_entry(goal_graph, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(goal_graph, ConnType.send, self._receive_goal_update))

        for actuator in self._directory.actuation:
            out_id_channels.append(self.sender_reg_entry(actuator, ConnType.request))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=HLOutbox
        )

    def _receive_belief_update(self, sender: str, channel: str, msg: Message) -> HLState:
        self.info(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.debug(f'Finished processing belief update {msg.id}!')
        return update

    def _receive_goal_update(self, sender: str, channel: str, msg: Message) -> HLState:
        self.info(f'Received goal update {msg.id} from {sender}. Processing...')
        update = self._base.on_goal_update(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing goal update {msg.id}!')
        return update
