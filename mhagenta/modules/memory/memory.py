from typing import Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Observation, Belief, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class MemoryOutbox(Outbox):
    """Internal communication outbox class for Memory structure.

    Used to store and process outgoing messages to other modules.

    """
    def send_observations(self, learner_id: str, observations: Iterable[Observation], **kwargs) -> None:
        """Send observation memories to a learner.

        Args:
            learner_id (str): `module_id` of the relevant learner.
            observations (Iterable[Observation]): A collection of observations to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'observations': observations}
        if kwargs:
            body.update(kwargs)
        self._add(learner_id, ConnType.send, body)

    def send_beliefs(self, hl_reasoner_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        """Send belief memories to a high-level reasoner.

        Args:
            hl_reasoner_id (str): `module_id` of the relevant high-level reasoner.
            beliefs (Iterable[Belief]): A collection of beliefs to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(hl_reasoner_id, ConnType.send, body)


MemoryState = State[MemoryOutbox]


class MemoryBase(ModuleBase):
    """Base class for defining Memory structure behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.MEMORY

    def on_observation_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        """Override to define memory structure's reaction to receiving a request for observation memories.

        Args:
            state (MemoryState): Memory structure's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the learner that sent the request.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            MemoryState: modified or unaltered internal state of the module.

        """
        return state

    def on_belief_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        """Override to define memory structure's reaction to receiving a request for belief memories.

        Args:
            state (MemoryState): Memory structure's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the knowledge model that sent the request.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            MemoryState: modified or unaltered internal state of the module.

        """
        return state

    def on_observation_update(self, state: MemoryState, sender: str, observations: Iterable[Observation], **kwargs) -> MemoryState:
        """Override to define memory structure's reaction to receiving an update of observation memories.

        Args:
            state (MemoryState): Memory structure's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the low-level reasoner that sent the update.
            observations (Iterable[Observation]): received collection of observations.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            MemoryState: modified or unaltered internal state of the module.

        """
        return state

    def on_belief_update(self, state: MemoryState, sender: str, beliefs: Iterable[Belief], **kwargs) -> MemoryState:
        """Override to define memory structure's reaction to receiving an update of belief memories.

        Args:
            state (MemoryState): Memory structure's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the knowledge model that sent the update.
            beliefs (Iterable[Belief]): received collection of beliefs.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            MemoryState: modified or unaltered internal state of the module.

        """
        return state


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
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=MemoryOutbox
        )

    def _receive_observations(self, sender: str, channel: str, msg: Message) -> MemoryState:
        self.info(f'Received observation update {msg.id} from {sender}. Processing...')
        observations = msg.body.pop('observations')
        update = self._base.on_observation_update(state=self._state, sender=sender, observations=observations, **msg.body)
        self.debug(f'Finished processing observation update {msg.id}!')
        return update

    def _receive_beliefs(self, sender: str, channel: str, msg: Message) -> MemoryState:
        self.info(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.debug(f'Finished processing belief update {msg.id}!')
        return update

    def _receive_observation_request(self, sender: str, channel: str, msg: Message) -> MemoryState:
        self.info(f'Received observation request {msg.id} from {sender}. Processing...')
        update = self._base.on_observation_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing observation request {msg.id}!')
        return update

    def _receive_belief_request(self, sender: str, channel: str, msg: Message) -> MemoryState:
        self.info(f'Received belief request {msg.id} from {sender}. Processing...')
        update = self._base.on_belief_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing belief request {msg.id}!')
        return update
