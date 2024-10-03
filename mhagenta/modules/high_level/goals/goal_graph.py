from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Goal, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class GoalGraphOutbox(Outbox):
    def send_goals(self, receiver: str, goals: Iterable[Goal], **kwargs) -> None:
        body = {'goals': goals}
        if kwargs:
            body.update(kwargs)
        self._add(receiver, ConnType.send, body)


GoalGraphState = State[GoalGraphOutbox]


class GoalGraphBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.GOALGRAPH

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs) -> GoalGraphState:
        raise NotImplementedError()

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Iterable[Goal], **kwargs) -> GoalGraphState:
        raise NotImplementedError()


class GoalGraph(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: GoalGraphBase
                 ) -> None:
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.request, self._receive_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.send, self._receive_update))

        for hl_reasoner in self._directory.hl_reasoning:
            out_id_channels.append(self.sender_reg_entry(hl_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner, ConnType.send, self._receive_update))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=GoalGraphOutbox
        )

    def _receive_update(self, sender: str, channel: str, msg: Message) -> GoalGraphState:
        self.info(f'Received goals update {msg.id} from {sender}. Processing...')
        goals = msg.body.pop('goals')
        update = self._base.on_goal_update(state=self._state, sender=sender, goals=goals, **msg.body)
        self.debug(f'Finished processing goal request {msg.id}!')
        return update

    def _receive_request(self, sender: str, channel: str, msg: Message) -> GoalGraphState:
        self.info(f'Received goals request {msg.id} from {sender}. Processing...')
        update = self._base.on_goal_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing goal request {msg.id}!')
        return update
