from typing import Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Goal, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class GoalGraphOutbox(Outbox):
    """Internal communication outbox class for Goal graph.

    Used to store and process outgoing messages to other modules.

    """
    def send_goals(self, receiver_id: str, goals: Iterable[Goal], **kwargs) -> None:
        """Update a low-level or a high-level reasoner on new or modified goals and their statuses.

        Args:
            receiver_id (str): `module_id` of the relevant low-level or high-level reasoner.
            goals (Iterable[Goal]): A collection of goals.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'goals': goals}
        if kwargs:
            body.update(kwargs)
        self._add(receiver_id, ConnType.send, body)


GoalGraphState = State[GoalGraphOutbox]


class GoalGraphBase(ModuleBase):
    """Base class for defining Goal graph behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.GOALGRAPH

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs) -> GoalGraphState:
        """Override to define goal graph's reaction to receiving a goal request.

        Args:
            state (GoalGraphState): Goal graph's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the module (high-level or low-level reasoner) that sent the request.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            GoalGraphState: modified or unaltered internal state of the module.

        """
        return state

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Iterable[Goal], **kwargs) -> GoalGraphState:
        """Override to define goal graph's reaction to receiving a goal update.

        Args:
            state (GoalGraphState): Goal graph's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the module (high-level or low-level reasoner) that sent the update.
            goals (Iterable[Goal]): received collection of goals.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            GoalGraphState: modified or unaltered internal state of the module.

        """
        return state


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
