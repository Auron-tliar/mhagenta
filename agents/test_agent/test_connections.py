import json
import dill
from typing import Any, Iterable, Self, Iterator, Callable, Literal
from pathlib import Path
import logging
import asyncio

from mhagenta import State, Observation, ActionStatus, Goal, Belief
from mhagenta.base import *
from mhagenta.outboxes import *
from mhagenta.utils.common import Directory
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes import MHARoot
from mhagenta.core import RabbitMQConnector


class TestData:
    _n_ll_reasoners = 1
    _n_perceptors = 1
    _n_actuators = 1
    _n_learners = 1
    _n_knowledge = 1
    _n_hl_reasoners = 1
    _n_goal_graphs = 1
    _n_memory = 1

    class ModuleIterator(Iterator):
        def __init__(self, module_id_template: str, total: int) -> None:
            self._module_id_template = module_id_template
            self._counter = -1
            self._total = total

        def __iter__(self) -> Self:
            self._counter = -1
            return self

        def __next__(self) -> str:
            self._counter += 1
            if self._counter >= self._total:
                raise StopIteration
            return self._module_id_template.format(self._counter)

    def __init__(self,
                 n_ll_reasoners: int | None = None,
                 n_perceptors: int | None = None,
                 n_actuators: int | None = None,
                 n_learners: int | None = None,
                 n_knowledge: int | None = None,
                 n_hl_reasoners: int | None = None,
                 n_goal_graphs: int | None = None,
                 n_memory: int | None = None,
                 save_format: Literal['json', 'dill'] = 'json'
                 ) -> None:
        self._ll_reasoners = self.ModuleIterator('test_ll_reasoner_{}', self._n_ll_reasoners if n_ll_reasoners is None else n_ll_reasoners)
        self._perceptors = self.ModuleIterator('test_perceptor_{}', self._n_perceptors if n_perceptors is None else n_perceptors)
        self._actuators = self.ModuleIterator('test_actuator_{}', self._n_actuators if n_actuators is None else n_actuators)
        self._learners = self.ModuleIterator('test_learner_{}', self._n_learners if n_learners is None else n_learners)
        self._knowledge = self.ModuleIterator('test_knowledge_{}', self._n_knowledge if n_knowledge is None else n_knowledge)
        self._hl_reasoners = self.ModuleIterator('test_hl_reasoner_{}', self._n_hl_reasoners if n_hl_reasoners is None else n_hl_reasoners)
        self._goal_graphs = self.ModuleIterator('test_goal_graph_{}', self._n_goal_graphs if n_goal_graphs is None else n_goal_graphs)
        self._memory = self.ModuleIterator('test_memory_{}', self._n_memory if n_memory is None else n_memory)

        self.save_format = save_format

    @property
    def ll_reasoners(self) -> Iterator:
        return self._ll_reasoners

    @property
    def perceptors(self) -> Iterator:
        return self._perceptors

    @property
    def actuators(self) -> Iterator:
        return self._actuators

    @property
    def learners(self) -> Iterator:
        return self._learners

    @property
    def knowledge(self) -> Iterator:
        return self._knowledge

    @property
    def hl_reasoner(self) -> Iterator:
        return self._hl_reasoners

    @property
    def goal_graphs(self) -> Iterator:
        return self._goal_graphs

    @property
    def memory(self) -> Iterator:
        return self._memory

    def expected_all(self) -> dict[str, dict[str, set]]:
        expected = dict()
        for ll_reasoner in self.ll_reasoners:
            expected[ll_reasoner] = self.expected_ll_reasoner(ll_reasoner)
        for perceptor in self.perceptors:
            expected[perceptor] = self.expected_perceptor(perceptor)
        for actuator in self.actuators:
            expected[actuator] = self.expected_actuator(actuator)
        for learner in self.learners:
            expected[learner] = self.expected_learner(learner)
        for knowledge in self.knowledge:
            expected[knowledge] = self.expected_knowledge(knowledge)
        for hl_reasoner in self.hl_reasoner:
            expected[hl_reasoner] = self.expected_hl_reasoner(hl_reasoner)
        for goal_graph in self.goal_graphs:
            expected[goal_graph] = self.expected_goal_graph(goal_graph)
        for memory in self.memory:
            expected[memory] = self.expected_memory(memory)

        return expected

    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_observation',
                'on_action_status',
                'on_goal_update',
                'on_learning_status',
                'on_model'
            },
            'received_from': set()
        }

        for perceptor in self.perceptors:
            expected['received_from'].update({
                ('on_observation', perceptor, 'step', module_id),
                ('on_observation', perceptor, 'on_request', module_id)
            })

        for actuator in self.actuators:
            expected['received_from'].update({
                ('on_action_status', actuator, 'step', module_id),
                ('on_action_status', actuator, 'on_request', module_id)
            })

        for goal_graph in self.goal_graphs:
            expected['received_from'].update({
                ('on_goal_update', goal_graph, 'step', module_id),
                ('on_goal_update', goal_graph, 'on_goal_update', module_id),
                ('on_goal_update', goal_graph, 'on_goal_request', module_id)
            })

        for learner in self.learners:
            expected['received_from'].update({
                ('on_learning_status', learner, 'step', module_id),
                ('on_learning_status', learner, 'on_task', module_id),
                ('on_learning_status', learner, 'on_memories', module_id),
                ('on_learning_status', learner, 'on_model_request', module_id),

                ('on_model', learner, 'step', module_id),
                ('on_model', learner, 'on_task', module_id),
                ('on_model', learner, 'on_memories', module_id),
                ('on_model', learner, 'on_model_request', module_id)
            })

        return expected

    def expected_perceptor(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_request'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_request', ll_reasoner, 'step', module_id),
                ('on_request', ll_reasoner, 'on_observation', module_id),
                ('on_request', ll_reasoner, 'on_action_status', module_id),
                ('on_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_request', ll_reasoner, 'on_learning_status', module_id),
                ('on_request', ll_reasoner, 'on_model', module_id)
            })

        return expected

    def expected_actuator(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_request'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_request', ll_reasoner, 'step', module_id),
                ('on_request', ll_reasoner, 'on_observation', module_id),
                ('on_request', ll_reasoner, 'on_action_status', module_id),
                ('on_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_request', ll_reasoner, 'on_learning_status', module_id),
                ('on_request', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoner:
            expected['received_from'].update({
                ('on_request', hl_reasoner, 'step', module_id),
                ('on_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_request', hl_reasoner, 'on_goal_update', module_id)
            })

        return expected

    def expected_learner(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_task',
                'on_memories',
                'on_model_request'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_task', ll_reasoner, 'step', module_id),
                ('on_task', ll_reasoner, 'on_observation', module_id),
                ('on_task', ll_reasoner, 'on_action_status', module_id),
                ('on_task', ll_reasoner, 'on_goal_update', module_id),
                ('on_task', ll_reasoner, 'on_learning_status', module_id),
                ('on_task', ll_reasoner, 'on_model', module_id),

                ('on_model_request', ll_reasoner, 'step', module_id),
                ('on_model_request', ll_reasoner, 'on_observation', module_id),
                ('on_model_request', ll_reasoner, 'on_action_status', module_id),
                ('on_model_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_model_request', ll_reasoner, 'on_learning_status', module_id),
                ('on_model_request', ll_reasoner, 'on_model', module_id)
            })

        for memory in self.memory:
            expected['received_from'].update({
                ('on_memories', memory, 'step', module_id),
                ('on_memories', memory, 'on_belief_request', module_id),
                ('on_memories', memory, 'on_belief_update', module_id),
                ('on_memories', memory, 'on_observation_request', module_id),
                ('on_memories', memory, 'on_observation_update', module_id)
            })

        return expected

    def expected_knowledge(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_belief_request',
                'on_belief_update'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_belief_update', ll_reasoner, 'step', module_id),
                ('on_belief_update', ll_reasoner, 'on_observation', module_id),
                ('on_belief_update', ll_reasoner, 'on_action_status', module_id),
                ('on_belief_update', ll_reasoner, 'on_goal_update', module_id),
                ('on_belief_update', ll_reasoner, 'on_learning_status', module_id),
                ('on_belief_update', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoner:
            expected['received_from'].update({
                ('on_belief_request', hl_reasoner, 'step', module_id),
                ('on_belief_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_belief_request', hl_reasoner, 'on_goal_update', module_id),

                ('on_belief_update', hl_reasoner, 'step', module_id),
                ('on_belief_update', hl_reasoner, 'on_belief_update', module_id),
                ('on_belief_update', hl_reasoner, 'on_goal_update', module_id)
            })

        return expected

    def expected_hl_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_belief_update',
                'on_goal_update'
            },
            'received_from': set()
        }

        for knowledge in self.knowledge:
            expected['received_from'].update({
                ('on_belief_update', knowledge, 'step', module_id),
                ('on_belief_update', knowledge, 'on_belief_request', module_id),
                ('on_belief_update', knowledge, 'on_belief_update', module_id),
            })

        for goal_graph in self.goal_graphs:
            expected['received_from'].update({
                ('on_goal_update', goal_graph, 'step', module_id),
                ('on_goal_update', goal_graph, 'on_goal_update', module_id),
                ('on_goal_update', goal_graph, 'on_goal_request', module_id)
            })

        for memory in self.memory:
            expected['received_from'].update({
                ('on_belief_update', memory, 'step', module_id),
                ('on_belief_update', memory, 'on_belief_request', module_id),
                ('on_belief_update', memory, 'on_belief_update', module_id),
                ('on_belief_update', memory, 'on_observation_request', module_id),
                ('on_belief_update', memory, 'on_observation_update', module_id)
            })

        return expected

    def expected_goal_graph(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_goal_update',
                'on_goal_request'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_goal_update', ll_reasoner, 'step', module_id),
                ('on_goal_update', ll_reasoner, 'on_observation', module_id),
                ('on_goal_update', ll_reasoner, 'on_action_status', module_id),
                ('on_goal_update', ll_reasoner, 'on_goal_update', module_id),
                ('on_goal_update', ll_reasoner, 'on_learning_status', module_id),
                ('on_goal_update', ll_reasoner, 'on_model', module_id),

                ('on_goal_request', ll_reasoner, 'step', module_id),
                ('on_goal_request', ll_reasoner, 'on_observation', module_id),
                ('on_goal_request', ll_reasoner, 'on_action_status', module_id),
                ('on_goal_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_goal_request', ll_reasoner, 'on_learning_status', module_id),
                ('on_goal_request', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoner:
            expected['received_from'].update({
                ('on_goal_update', hl_reasoner, 'step', module_id),
                ('on_goal_update', hl_reasoner, 'on_belief_update', module_id),
                ('on_goal_update', hl_reasoner, 'on_goal_update', module_id)
            })

        return expected

    def expected_memory(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_belief_request',
                'on_belief_update',
                'on_observation_request',
                'on_observation_update'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_observation_update', ll_reasoner, 'step', module_id),
                ('on_observation_update', ll_reasoner, 'on_observation', module_id),
                ('on_observation_update', ll_reasoner, 'on_action_status', module_id),
                ('on_observation_update', ll_reasoner, 'on_goal_update', module_id),
                ('on_observation_update', ll_reasoner, 'on_learning_status', module_id),
                ('on_observation_update', ll_reasoner, 'on_model', module_id)
            })

        for learner in self.learners:
            expected['received_from'].update({
                ('on_observation_request', learner, 'step', module_id),
                ('on_observation_request', learner, 'on_task', module_id),
                ('on_observation_request', learner, 'on_memories', module_id),
                ('on_observation_request', learner, 'on_model_request', module_id)
            })

        for knowledge in self.knowledge:
            expected['received_from'].update({
                ('on_belief_update', knowledge, 'step', module_id),
                ('on_belief_update', knowledge, 'on_belief_request', module_id),
                ('on_belief_update', knowledge, 'on_belief_update', module_id),
            })

        for hl_reasoner in self.hl_reasoner:
            expected['received_from'].update({
                ('on_belief_request', hl_reasoner, 'step', module_id),
                ('on_belief_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_belief_request', hl_reasoner, 'on_goal_update', module_id)
            })

        return expected


class BaseAuxiliary:
    @staticmethod
    def _signature_gen_factory(module_id: str, func_name: str) -> Callable[[str], tuple[str, str, str]]:
        def signature_gen(recipient: str) -> tuple[str, str, str]:
            return module_id, func_name, recipient

        return signature_gen

    def process_and_send(self, func_name: str, state: State, signature: tuple[str, str, str] | None = None) -> Update:
        if signature is not None:
            state.received_from.add((func_name, signature[0], signature[1], signature[2]))
        if func_name not in state.sent_from:
            outbox = self.message_all(state.directory, func_name)
            state.sent_from.add(func_name)
        else:
            outbox = None
        return state, outbox


class TestLLReasoner(LLReasonerBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_observation(self, state: State, sender: str, observation: Observation, **kwargs) -> Update:
        return self.process_and_send(self.on_observation.__name__, state, kwargs['signature'])

    def on_action_status(self, state: State, sender: str, action_status: ActionStatus, **kwargs) -> Update:
        return self.process_and_send(self.on_action_status.__name__, state, kwargs['signature'])

    def on_goal_update(self, state: State, sender: str, goals: list[Goal], **kwargs) -> Update:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def on_learning_status(self, state: State, sender: str, learning_status: Any, **kwargs) -> Update:
        return self.process_and_send(self.on_learning_status.__name__, state, kwargs['signature'])

    def on_model(self, state: State, sender: str, model: Any, **kwargs) -> Update:
        return self.process_and_send(self.on_model.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> LLOutbox:
        outbox = LLOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator, signature=signature_gen(actuator))
        for perceptor in directory.perception:
            outbox.request_observation(perceptor_id=perceptor, signature=signature_gen(perceptor))
        for learner in directory.learning:
            outbox.request_model(learner_id=learner, signature=signature_gen(learner))
            outbox.send_learner_task(learner_id=learner, task=None, signature=signature_gen(learner))
        for goal_graph in directory.goals:
            outbox.request_goals(goal_graph_id=goal_graph, signature=signature_gen(goal_graph))
            outbox.send_goal_update(goal_graph_id=goal_graph, goals=[], signature=signature_gen(goal_graph))
        for knowledge in directory.knowledge:
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[], signature=signature_gen(knowledge))
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, observations=[], signature=signature_gen(memory))
        return outbox


class TestActuator(ActuatorBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_request.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> ActuatorOutbox:
        outbox = ActuatorOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, status=ActionStatus(status=None), signature=signature_gen(ll_reasoner))
        return outbox


class TestPerceptor(PerceptorBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_request.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> PerceptorOutbox:
        outbox = PerceptorOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_observation(ll_reasoner_id=ll_reasoner, observation=Observation(observation_type='none', value=None), signature=signature_gen(ll_reasoner))
        return outbox


class TestLearner(LearnerBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_task(self, state: State, sender: str, task: Any, **kwargs) -> Update:
        return self.process_and_send(self.on_task.__name__, state, kwargs['signature'])

    def on_memories(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        return self.process_and_send(self.on_memories.__name__, state, kwargs['signature'])

    def on_model_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_model_request.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> LearnerOutbox:
        outbox = LearnerOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory, signature=signature_gen(memory))
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, learning_status=None, signature=signature_gen(ll_reasoner))
            outbox.send_model(ll_reasoner_id=ll_reasoner, model=None, signature=signature_gen(ll_reasoner))
        return outbox


class TestKnowledge(KnowledgeBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_belief_request.__name__, state, kwargs['signature'])

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> KnowledgeOutbox:
        outbox = KnowledgeOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(knowledge_id=hl_reasoner, beliefs=[], signature=signature_gen(hl_reasoner))
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, beliefs=[], signature=signature_gen(memory))
        return outbox


class TestHLReasoner(HLReasonerBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def on_goal_update(self, state: State, sender: str, goals: Iterable[Goal], **kwargs) -> Update:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> HLOutbox:
        outbox = HLOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory, signature=signature_gen(memory))
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator, signature=signature_gen(actuator))
        for knowledge in directory.knowledge:
            outbox.request_beliefs(knowledge_id=knowledge, signature=signature_gen(knowledge))
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[], signature=signature_gen(knowledge))
        for goal_graph in directory.goals:
            outbox.send_goals(goal_graph_id=goal_graph, goals=[], signature=signature_gen(goal_graph))
        return outbox


class TestGoalGraph(GoalGraphBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_goal_update(self, state: State, sender: str, goals: Iterable[Goal], **kwargs) -> Update:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def on_goal_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_goal_request.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> GoalGraphOutbox:
        outbox = GoalGraphOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_goals(receiver=hl_reasoner, goals=[], signature=signature_gen(hl_reasoner))
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_goals(receiver=ll_reasoner, goals=[], signature=signature_gen(ll_reasoner))
        return outbox


class TestMemory(MemoryBase, BaseAuxiliary):
    def step(self, state: State) -> Update:
        return self.process_and_send(self.step.__name__, state)

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_belief_request.__name__, state, kwargs['signature'])

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def on_observation_request(self, state: State, sender: str, **kwargs) -> Update:
        return self.process_and_send(self.on_observation_request.__name__, state, kwargs['signature'])

    def on_observation_update(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        return self.process_and_send(self.on_observation_update.__name__, state, kwargs['signature'])

    def message_all(self, directory: Directory, func_name: str) -> MemoryOutbox:
        outbox = MemoryOutbox()
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(hl_reasoner_id=hl_reasoner, beliefs=[], signature=signature_gen(hl_reasoner))
        for learner in directory.learning:
            outbox.send_observations(learner_id=learner, observations=[], signature=signature_gen(learner))
        return outbox


def check_module_result(module_id: str, actual_state: dict[str, set], expected_state: dict[str, set]) -> int:
    print(f'\t\t{module_id}...')
    result = (actual_state == expected_state)
    if result:
        print('\t\t\t...SUCCEEDED!')
        return 1
    else:
        print('\t\t\t...FAILED!')
        if actual_state['sent_from'] != expected_state['sent_from']:
            diff = actual_state['sent_from'].difference(expected_state['sent_from'])
            if diff:
                print('\t\t\t\tUnexpected \'sent from\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t".join([str(item) for item in diff])}')
            diff = expected_state['sent_from'].difference(actual_state['sent_from'])
            if diff:
                print('\t\t\t\tMissing \'sent_from\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t".join([str(item) for item in diff])}')
        if actual_state['received_from'] != expected_state['received_from']:
            diff = actual_state['received_from'].difference(expected_state['received_from'])
            if diff:
                print('\t\t\t\tUnexpected \'received from\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t".join([str(item) for item in diff])}')
            diff = expected_state['received_from'].difference(actual_state['received_from'])
            if diff:
                print('\t\t\t\tMissing \'received\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t".join([str(item) for item in diff])}')
        print()
        return 0


def check_module(agent_id: str, module_id: str, save_dir: Path, expected_state: dict[str, set], save_format: Literal['json', 'dill'] = 'json') -> int:
    save_format = test_data.save_format
    with open(Path(save_dir) / f'{agent_id}.{module_id}.{"sav" if save_format == "dill" else "json"}', 'rb') as f:
        actual_state = (dill if save_format == 'dill' else json).load(f)
    return check_module_result(module_id, actual_state, expected_state)


def check_results(agent_id: str, save_dir: str | Path, test_data: TestData) -> None:
    print(f'================== Checking final states... ==================')
    total_tests = 0
    successful = 0

    print('\tPerceptors:')
    for perceptor in test_data.perceptors:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=perceptor,
            save_dir=save_dir,
            expected_state=test_data.expected_perceptor(perceptor),
            save_format='dill')

    print('\tActuators:')
    for actuator in test_data.actuators:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=actuator,
            save_dir=save_dir,
            expected_state=test_data.expected_actuator(actuator),
            save_format='dill')

    print('\tLow-level reasoners:')
    for ll_reasoner in test_data.ll_reasoners:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=ll_reasoner,
            save_dir=save_dir,
            expected_state=test_data.expected_ll_reasoner(ll_reasoner),
            save_format='dill')

    print('\tLearners:')
    for learner in test_data.learners:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=learner,
            save_dir=save_dir,
            expected_state=test_data.expected_learner(learner),
            save_format='dill')

    print('\tKnowledge:')
    for knowledge in test_data.knowledge:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=knowledge,
            save_dir=save_dir,
            expected_state=test_data.expected_knowledge(knowledge),
            save_format='dill')

    print('\tHigh-level reasoners:')
    for hl_reasoner in test_data.hl_reasoner:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=hl_reasoner,
            save_dir=save_dir,
            expected_state=test_data.expected_hl_reasoner(hl_reasoner),
            save_format='dill')

    print('\tGoal graphs:')
    for goal_graph in test_data.goal_graphs:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=goal_graph,
            save_dir=save_dir,
            expected_state=test_data.expected_goal_graph(goal_graph),
            save_format='dill')

    print('\tMemory:')
    for memory in test_data.memory:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=memory,
            save_dir=save_dir,
            expected_state=test_data.expected_memory(memory),
            save_format='dill')

    print(f'TEST RESULTS: {successful}/{total_tests}')


async def run_agent(agent: MHARoot, test_data: TestData, only_test: bool = False) -> None:
    if not only_test:
        print('INITIALIZING...')
        await agent.initialize()
        print('INITIALIZED!\nRUNNING...')
        await agent.start()
        print('EXECUTION FINISHED!')
    print('CHECKING RESULTS...')
    check_results(agent_id=agent.agent_id, save_dir=agent.save_dir,test_data=test_data)


if __name__ == '__main__':
    test_data = TestData(save_format='dill')

    test_agent = MHARoot(
        agent_id='test_agent',
        connector_cls=RabbitMQConnector,
        perceptors=[TestPerceptor(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.perceptors],
        actuators=[TestActuator(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.actuators],
        ll_reasoners=[TestLLReasoner(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.ll_reasoners],
        learners=[TestLearner(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.learners],
        knowledge=[TestKnowledge(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.knowledge],
        hl_reasoner=[TestHLReasoner(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.hl_reasoner],
        goal_graphs=[TestGoalGraph(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.goal_graphs],
        memory=[TestMemory(
            module_id=module_id,
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.memory],
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=.5,
        exec_duration=10.,
        save_dir=Path('D:\\bsc-tmp\\phd\\phd_thesis\\hybrid-agent\\out\\test_agent_refactor\\connections'),
        save_format=test_data.save_format,
        resume=False,
        log_level=logging.DEBUG
    )

    asyncio.run(run_agent(agent=test_agent, test_data=test_data, only_test=False))
