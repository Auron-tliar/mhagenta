from typing import Iterable, ClassVar, Any

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Goal, Belief, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class HLOutbox(Outbox):
    """Internal communication outbox class for High-level reasoner.

    Used to store and process outgoing messages to other modules.

    """
    def request_beliefs(self, knowledge_id: str, **kwargs) -> None:
        """Request beliefs from a knowledge model.

        Args:
            knowledge_id (str): `module_id` of the relevant knowledge model.
            **kwargs: additional keyword arguments to be included in the message.

        """
        self._add(knowledge_id, ConnType.request, kwargs)

    def request_action(self, actuator_id: str, **kwargs) -> None:
        """Request an action from an actuator.

        Args:
            actuator_id (str): `module_id` of the actuator chosen to perform the action.
            **kwargs: additional keyword arguments to be included in the message.

        """
        self._add(actuator_id, ConnType.request, kwargs)

    def send_beliefs(self, knowledge_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        """Send new or updated beliefs to a knowledge model.

        Args:
            knowledge_id (str): `module_id` of the relevant knowledge model.
            beliefs (Iterable[Belief]): a collection of beliefs to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(knowledge_id, ConnType.send, body)

    def send_goals(self, goal_graph_id: str, goals: Iterable[Goal], **kwargs) -> None:
        """Send new or updated goals to a goal graph.

        Args:
            goal_graph_id (str): `module_id` of the relevant goal graph.
            goals (Iterable[Goal]): a collection of goals to send.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'goals': goals}
        if kwargs:
            body.update(kwargs)
        self._add(goal_graph_id, ConnType.send, body)

    def request_model(self, learner_id: str, **kwargs) -> None:
        """Request the current model from a learner.

        Args:
            learner_id (str): `module_id` of a learner training the required model.
            **kwargs: additional keyword arguments to be included in the message.

        """
        self._add(learner_id, ConnType.request, kwargs)

    def send_learner_task(self, learner_id: str, task: Any, **kwargs) -> None:
        """Send a new or updated learning task to a learner.

        Args:
            learner_id (str): `module_id` of the relevant learner.
            task (Any): an object specifying the learning task.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'task': task}
        if kwargs:
            body.update(kwargs)
        self._add(learner_id, ConnType.send, body)


HLState = State[HLOutbox]


class HLReasonerBase(ModuleBase):
    """Base class for defining High-level reasoner behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.HLREASONER

    def on_belief_update(self, state: HLState, sender: str, beliefs: Iterable[Belief], **kwargs) -> HLState:
        """Override to define high-level reasoner's reaction to receiving a belief update.

        Args:
            state (HLState): High-level reasoner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the Knowledge model that sent the update.
            beliefs (Iterable[Belief]): received collection of beliefs.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            HLState: modified or unaltered internal state of the module.

        """
        return state

    def on_goal_update(self, state: HLState, sender: str, goals: Iterable[Goal], **kwargs) -> HLState:
        """Override to define high-level reasoner's reaction to receiving a goal update.

        Args:
            state: High-level reasoner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the goal graph that sent the update.
            goals (Iterable[Goal]): received collection of goals.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            HLState: modified or unaltered internal state of the module.

        """
        return state

    def on_model(self, state: HLState, sender: str, model: Any, **kwargs) -> HLState:
        """Override to define high-level reasoner's reaction to receiving a learned model.

        Args:
            state (HLState): High-level reasoner's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the learner that sent the model.
            model (Any): received learned model object.
            **kwargs: additional keyword arguments included in the message.

        Returns:
            LLState: modified or unaltered internal state of the module.

        """
        return state


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

        for knowledge in self._directory.internal.knowledge:
            out_id_channels.append(self.sender_reg_entry(knowledge.module_id, ConnType.request))
            out_id_channels.append(self.sender_reg_entry(knowledge.module_id, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(knowledge.module_id, ConnType.send, self._receive_belief_update))

        for goal_graph in self._directory.internal.goals:
            out_id_channels.append(self.sender_reg_entry(goal_graph.module_id, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(goal_graph.module_id, ConnType.send, self._receive_goal_update))

        for actuator in self._directory.internal.actuation:
            out_id_channels.append(self.sender_reg_entry(actuator.module_id, ConnType.request))

        for learner in self._directory.internal.learning:
            out_id_channels.append(self.sender_reg_entry(learner.module_id, ConnType.request))
            out_id_channels.append(self.sender_reg_entry(learner.module_id, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(learner.module_id, ConnType.send, self._receive_learner_model))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=HLOutbox
        )

    def _receive_belief_update(self, sender: str, channel: str, msg: Message) -> HLState:
        self.debug(f'Received belief update {msg.id} from {sender}. Processing...')
        beliefs = msg.body.pop('beliefs')
        update = self._base.on_belief_update(state=self._state, sender=sender, beliefs=beliefs, **msg.body)
        self.log(5, f'Finished processing belief update {msg.id}!')
        return update

    def _receive_goal_update(self, sender: str, channel: str, msg: Message) -> HLState:
        self.debug(f'Received goal update {msg.id} from {sender}. Processing...')
        update = self._base.on_goal_update(state=self._state, sender=sender, **msg.body)
        self.log(5, f'Finished processing goal update {msg.id}!')
        return update

    def _receive_learner_model(self, sender: str, channel: str, msg: Message) -> HLState:
        self.debug(f'Received learned model {msg.id} from {sender}. Processing...')
        model = msg.body.pop('model')
        update = self._base.on_model(state=self._state, sender=sender, model=model, **msg.body)
        self.log(5, f'Finished processing learned model {msg.id}!')
        return update
