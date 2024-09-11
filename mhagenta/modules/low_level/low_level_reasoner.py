from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Goal, Observation, ActionStatus, Belief, State
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class LLOutbox(Outbox):
    def request_action(self, actuator_id: str = ModuleTypes.ACTUATOR, **kwargs) -> None:
        self._add(actuator_id, ConnType.request, kwargs)

    def request_observation(self, perceptor_id: str = ModuleTypes.PERCEPTOR, **kwargs) -> None:
        self._add(perceptor_id, ConnType.request, kwargs)

    def send_beliefs(self, knowledge_id: str, beliefs: Iterable[Belief], **kwargs) -> None:
        body = {'beliefs': beliefs}
        if kwargs:
            body.update(kwargs)
        self._add(knowledge_id, ConnType.send, body)

    def request_goals(self, goal_graph_id: str, **kwargs) -> None:
        self._add(goal_graph_id, ConnType.request, kwargs)

    def send_goal_update(self, goal_graph_id: str, goals: Iterable[Goal], **kwargs) -> None:
        body = {'goals': goals}
        if kwargs:
            body.update(kwargs)
        self._add(goal_graph_id, ConnType.send, body)

    def send_memories(self, memory_id: str, observations: Iterable[Any], **kwargs) -> None:
        body = {'observations': observations}
        if kwargs:
            body.update(kwargs)
        self._add(memory_id, ConnType.send, body)

    def request_model(self, learner_id: str, **kwargs) -> None:
        self._add(learner_id, ConnType.request, kwargs)

    def send_learner_task(self, learner_id: str, task: Any, **kwargs) -> None:
        body = {'task': task}
        if kwargs:
            body.update(kwargs)
        self._add(learner_id, ConnType.send, body)


class LLReasonerBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.LLREASONER

    def on_observation(self, state: State, sender: str, observation: Observation, **kwargs) -> Update:
        raise NotImplementedError()

    def on_action_status(self, state: State, sender: str, action_status: ActionStatus, **kwargs) -> Update:
        raise NotImplementedError()

    def on_goal_update(self, state: State, sender: str, goals: list[Goal], **kwargs) -> Update:
        raise NotImplementedError()

    def on_model(self, state: State, sender: str, model: Any, **kwargs) -> Update:
        raise NotImplementedError()

    def on_learning_status(self, state: State, sender: str, learning_status: Any, **kwargs) -> Update:
        raise NotImplementedError()


class LLReasoner(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: LLReasonerBase
                 ) -> None:
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for perceptor in self._directory.perception:
            out_id_channels.append(self.sender_reg_entry(perceptor, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(perceptor, ConnType.send, self._receive_observation))

        for actuator in self._directory.actuation:
            out_id_channels.append(self.sender_reg_entry(actuator, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(actuator, ConnType.send, self._receive_action_status))

        if self._directory.hl_reasoning:
            for knowledge in self._directory.knowledge:
                out_id_channels.append(self.sender_reg_entry(knowledge, ConnType.send))
            for goal_graph in self._directory.goals:
                out_id_channels.append(self.sender_reg_entry(goal_graph, ConnType.request))
                out_id_channels.append(self.sender_reg_entry(goal_graph, ConnType.send))
                in_id_channels_callbacks.append(self.recipient_reg_entry(goal_graph, ConnType.send, self._receive_goals))
        for memory in self._directory.memory:
            out_id_channels.append(self.sender_reg_entry(memory, ConnType.send))
        for learner in self._directory.learning:
            out_id_channels.append(self.sender_reg_entry(learner, ConnType.request))
            out_id_channels.append(self.sender_reg_entry(learner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(learner, ConnType.send, self._receive_learning_status, 'status'))
            in_id_channels_callbacks.append(self.recipient_reg_entry(learner, ConnType.send, self._receive_learner_model, 'model'))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks
        )

    def _receive_observation(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received observation {msg.id} from {sender}. Processing...')
        observation = msg.body.pop('observation')
        update = self._base.on_observation(state=self._state, sender=sender, observation=observation, **msg.body)
        self.debug(f'Finished processing observation {msg.id}!')
        return update

    def _receive_action_status(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received action status {msg.id} from {sender}. Processing...')
        action_status = msg.body.pop('action_status')
        update = self._base.on_action_status(state=self._state, sender=sender, action_status=action_status, **msg.body)
        self.debug(f'Finished processing action status {msg.id}!')
        return update

    def _receive_goals(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received goal update {msg.id} from {sender}. Processing...')
        goals = msg.body.pop('goals')
        update = self._base.on_goal_update(state=self._state, sender=sender, goals=goals, **msg.body)
        self.debug(f'Finished processing goal update {msg.id}!')
        return update

    def _receive_learning_status(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received learning status {msg.id} from {sender}. Processing...')
        learning_status = msg.body.pop('learning_status')
        update = self._base.on_learning_status(state=self._state, sender=sender, learning_status=learning_status, **msg.body)
        self.debug(f'Finished processing learning status {msg.id}!')
        return update

    def _receive_learner_model(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received learned model {msg.id} from {sender}. Processing...')
        model = msg.body.pop('model')
        update = self._base.on_model(state=self._state, sender=sender, model=model, **msg.body)
        self.debug(f'Finished processing learned model {msg.id}!')
        return update
