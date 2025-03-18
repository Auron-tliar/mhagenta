import json
import sys

import dill
from typing import Any, Iterable, Self, Iterator, Callable, Literal
from pathlib import Path
import logging
import asyncio
from sys import argv

from mhagenta import State, Observation, ActionStatus, Goal, Belief, Orchestrator
from mhagenta.bases import *
from mhagenta.states import *
from mhagenta.utils.common import Directory
from mhagenta.core import RabbitMQConnector

from base import TestDataBase, check_results


class TestData(TestDataBase):
    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_observation',
                'on_action_status',
                'on_goal_update',
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
                ('on_request', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoners:
            expected['received_from'].update({
                ('on_request', hl_reasoner, 'step', module_id),
                ('on_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_request', hl_reasoner, 'on_goal_update', module_id),
                ('on_request', hl_reasoner, 'on_model', module_id)
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
                ('on_task', ll_reasoner, 'on_model', module_id),

                ('on_model_request', ll_reasoner, 'step', module_id),
                ('on_model_request', ll_reasoner, 'on_observation', module_id),
                ('on_model_request', ll_reasoner, 'on_action_status', module_id),
                ('on_model_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_model_request', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoners:
            expected['received_from'].update({
                ('on_task', hl_reasoner, 'step', module_id),
                ('on_task', hl_reasoner, 'on_belief_update', module_id),
                ('on_task', hl_reasoner, 'on_goal_update', module_id),
                ('on_task', hl_reasoner, 'on_model', module_id),

                ('on_model_request', hl_reasoner, 'step', module_id),
                ('on_model_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_model_request', hl_reasoner, 'on_goal_update', module_id),
                ('on_model_request', hl_reasoner, 'on_model', module_id)
            })

        for memory in self.memory:
            expected['received_from'].update({
                ('on_memories', memory, 'step', module_id),
                ('on_memories', memory, 'on_memory_request', module_id),
                ('on_memories', memory, 'on_belief_update', module_id),
                ('on_memories', memory, 'on_observation_update', module_id)
            })

        return expected

    def expected_knowledge(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_belief_request',
                'on_belief_update',
                'on_observed_beliefs'
            },
            'received_from': set()
        }

        for ll_reasoner in self.ll_reasoners:
            expected['received_from'].update({
                ('on_observed_beliefs', ll_reasoner, 'step', module_id),
                ('on_observed_beliefs', ll_reasoner, 'on_observation', module_id),
                ('on_observed_beliefs', ll_reasoner, 'on_action_status', module_id),
                ('on_observed_beliefs', ll_reasoner, 'on_goal_update', module_id),
                ('on_observed_beliefs', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoners:
            expected['received_from'].update({
                ('on_belief_request', hl_reasoner, 'step', module_id),
                ('on_belief_request', hl_reasoner, 'on_belief_update', module_id),
                ('on_belief_request', hl_reasoner, 'on_goal_update', module_id),
                ('on_belief_request', hl_reasoner, 'on_model', module_id),

                ('on_belief_update', hl_reasoner, 'step', module_id),
                ('on_belief_update', hl_reasoner, 'on_belief_update', module_id),
                ('on_belief_update', hl_reasoner, 'on_goal_update', module_id),
                ('on_belief_update', hl_reasoner, 'on_model', module_id)
            })

        return expected

    def expected_hl_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_belief_update',
                'on_goal_update',
                'on_model'
            },
            'received_from': set()
        }

        for knowledge in self.knowledge:
            expected['received_from'].update({
                ('on_belief_update', knowledge, 'step', module_id),
                ('on_belief_update', knowledge, 'on_belief_request', module_id),
                ('on_belief_update', knowledge, 'on_belief_update', module_id),
                ('on_belief_update', knowledge, 'on_observed_beliefs', module_id)
            })

        for goal_graph in self.goal_graphs:
            expected['received_from'].update({
                ('on_goal_update', goal_graph, 'step', module_id),
                ('on_goal_update', goal_graph, 'on_goal_update', module_id),
                ('on_goal_update', goal_graph, 'on_goal_request', module_id)
            })

        for learner in self.learners:
            expected['received_from'].update({
                ('on_model', learner, 'step', module_id),
                ('on_model', learner, 'on_task', module_id),
                ('on_model', learner, 'on_memories', module_id),
                ('on_model', learner, 'on_model_request', module_id)
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
                ('on_goal_update', ll_reasoner, 'on_model', module_id),

                ('on_goal_request', ll_reasoner, 'step', module_id),
                ('on_goal_request', ll_reasoner, 'on_observation', module_id),
                ('on_goal_request', ll_reasoner, 'on_action_status', module_id),
                ('on_goal_request', ll_reasoner, 'on_goal_update', module_id),
                ('on_goal_request', ll_reasoner, 'on_model', module_id)
            })

        for hl_reasoner in self.hl_reasoners:
            expected['received_from'].update({
                ('on_goal_update', hl_reasoner, 'step', module_id),
                ('on_goal_update', hl_reasoner, 'on_belief_update', module_id),
                ('on_goal_update', hl_reasoner, 'on_goal_update', module_id),
                ('on_goal_update', hl_reasoner, 'on_model', module_id)
            })

        return expected

    def expected_memory(self, module_id: str) -> dict[str, set]:
        expected = {
            'sent_from': {
                'step',
                'on_memory_request',
                'on_belief_update',
                'on_observation_update'
            },
            'received_from': set()
        }

        for learner in self.learners:
            expected['received_from'].update({
                ('on_memory_request', learner, 'step', module_id),
                ('on_memory_request', learner, 'on_task', module_id),
                ('on_memory_request', learner, 'on_memories', module_id),
                ('on_memory_request', learner, 'on_model_request', module_id)
            })

        for knowledge in self.knowledge:
            expected['received_from'].update({
                ('on_belief_update', knowledge, 'step', module_id),
                ('on_belief_update', knowledge, 'on_belief_request', module_id),
                ('on_belief_update', knowledge, 'on_belief_update', module_id),
                ('on_belief_update', knowledge, 'on_observed_beliefs', module_id),

                ('on_observation_update', knowledge, 'step', module_id),
                ('on_observation_update', knowledge, 'on_belief_request', module_id),
                ('on_observation_update', knowledge, 'on_belief_update', module_id),
                ('on_observation_update', knowledge, 'on_observed_beliefs', module_id)
            })

        return expected


class BaseAuxiliary:
    @staticmethod
    def _signature_gen_factory(module_id: str, func_name: str) -> Callable[[str], tuple[str, str, str]]:
        def signature_gen(recipient: str) -> tuple[str, str, str]:
            return module_id, func_name, recipient

        return signature_gen

    def process_and_send(self, func_name: str, state: State, signature: tuple[str, str, str] | None = None) -> State:
        if signature is not None:
            state.received_from.add((func_name, signature[0], signature[1], signature[2]))
        if func_name not in state.sent_from:
            self.message_all(state, state.directory, func_name)
            state.sent_from.add(func_name)
        return state

    def message_all(self, state: State, directory: str, func_name: str) -> None:
        pass


class TestLLReasoner(LLReasonerBase, BaseAuxiliary):
    def step(self, state: LLState) -> LLState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs) -> LLState:
        return self.process_and_send(self.on_observation.__name__, state, kwargs['signature'])

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs) -> LLState:
        return self.process_and_send(self.on_action_status.__name__, state, kwargs['signature'])

    def on_goal_update(self, state: LLState, sender: str, goals: list[Goal], **kwargs) -> LLState:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs) -> LLState:
        return self.process_and_send(self.on_model.__name__, state, kwargs['signature'])

    def message_all(self, state: LLState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for actuator in directory.internal.actuation:
            state.outbox.request_action(actuator_id=actuator.module_id, signature=signature_gen(actuator.module_id))
        for perceptor in directory.internal.perception:
            state.outbox.request_observation(perceptor_id=perceptor.module_id, signature=signature_gen(perceptor.module_id))
        for learner in directory.internal.learning:
            state.outbox.request_model(learner_id=learner.module_id, signature=signature_gen(learner.module_id))
            state.outbox.send_learner_task(learner_id=learner.module_id, task=None, signature=signature_gen(learner.module_id))
        for goal_graph in directory.internal.goals:
            state.outbox.request_goals(goal_graph_id=goal_graph.module_id, signature=signature_gen(goal_graph.module_id))
            state.outbox.send_goal_update(goal_graph_id=goal_graph.module_id, goals=[], signature=signature_gen(goal_graph.module_id))
        for knowledge in directory.internal.knowledge:
            state.outbox.send_beliefs(knowledge_id=knowledge.module_id, observation=Observation(observation_type='none', content=None), beliefs=[], signature=signature_gen(knowledge.module_id))


class TestActuator(ActuatorBase, BaseAuxiliary):
    def step(self, state: ActuatorState) -> ActuatorState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        return self.process_and_send(self.on_request.__name__, state, kwargs['signature'])

    def message_all(self, state: ActuatorState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for ll_reasoner in directory.internal.ll_reasoning:
            state.outbox.send_status(ll_reasoner_id=ll_reasoner.module_id, status=ActionStatus(status=None), signature=signature_gen(ll_reasoner.module_id))


class TestPerceptor(PerceptorBase, BaseAuxiliary):
    def step(self, state: PerceptorState) -> PerceptorState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_request(self, state: PerceptorState, sender: str, **kwargs) -> PerceptorState:
        return self.process_and_send(self.on_request.__name__, state, kwargs['signature'])

    def message_all(self, state: PerceptorState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for ll_reasoner in directory.internal.ll_reasoning:
            state.outbox.send_observation(ll_reasoner_id=ll_reasoner.module_id, observation=Observation(observation_type='none', content=None), signature=signature_gen(ll_reasoner.module_id))


class TestLearner(LearnerBase, BaseAuxiliary):
    def step(self, state: LearnerState) -> LearnerState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs) -> LearnerState:
        return self.process_and_send(self.on_task.__name__, state, kwargs['signature'])

    def on_memories(self, state: LearnerState, sender: str, memories: Iterable[Belief | Observation], **kwargs) -> LearnerState:
        return self.process_and_send(self.on_memories.__name__, state, kwargs['signature'])

    def on_model_request(self, state: LearnerState, sender: str, **kwargs) -> LearnerState:
        return self.process_and_send(self.on_model_request.__name__, state, kwargs['signature'])

    def message_all(self, state: LearnerState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for memory in directory.internal.memory:
            state.outbox.request_memories(memory_id=memory.module_id, signature=signature_gen(memory.module_id))
        for ll_reasoner in directory.internal.ll_reasoning:
            state.outbox.send_model(reasoner_id=ll_reasoner.module_id, model=None, signature=signature_gen(ll_reasoner.module_id))
        for hl_reasoner in directory.internal.hl_reasoning:
            state.outbox.send_model(reasoner_id=hl_reasoner.module_id, model=None, signature=signature_gen(hl_reasoner.module_id))


class TestKnowledge(KnowledgeBase, BaseAuxiliary):
    def step(self, state: KnowledgeState) -> KnowledgeState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs) -> KnowledgeState:
        return self.process_and_send(self.on_belief_request.__name__, state, kwargs['signature'])

    def on_belief_update(self, state: KnowledgeState, sender: str, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        return self.process_and_send(self.on_observed_beliefs.__name__, state, kwargs['signature'])

    def message_all(self, state: KnowledgeState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for hl_reasoner in directory.internal.hl_reasoning:
            state.outbox.send_beliefs(hl_reasoner_id=hl_reasoner.module_id, beliefs=[], signature=signature_gen(hl_reasoner.module_id))
        for memory in directory.internal.memory:
            state.outbox.send_belief_memories(memory_id=memory.module_id, beliefs=[], signature=signature_gen(memory.module_id))
            state.outbox.send_observations(memory_id=memory.module_id, observations=[], signature=signature_gen(memory.module_id))


class TestHLReasoner(HLReasonerBase, BaseAuxiliary):
    def step(self, state: HLState) -> HLState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_belief_update(self, state: HLState, sender: str, beliefs: Iterable[Belief], **kwargs) -> HLState:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def on_goal_update(self, state: HLState, sender: str, goals: Iterable[Goal], **kwargs) -> HLState:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def on_model(self, state: HLState, sender: str, model: Any, **kwargs) -> HLState:
        return self.process_and_send(self.on_model.__name__, state, kwargs['signature'])

    def message_all(self, state: HLState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for actuator in directory.internal.actuation:
            state.outbox.request_action(actuator_id=actuator.module_id, signature=signature_gen(actuator.module_id))
        for knowledge in directory.internal.knowledge:
            state.outbox.request_beliefs(knowledge_id=knowledge.module_id, signature=signature_gen(knowledge.module_id))
            state.outbox.send_beliefs(knowledge_id=knowledge.module_id, beliefs=[], signature=signature_gen(knowledge.module_id))
        for goal_graph in directory.internal.goals:
            state.outbox.send_goals(goal_graph_id=goal_graph.module_id, goals=[], signature=signature_gen(goal_graph.module_id))
        for learner in directory.internal.learning:
            state.outbox.request_model(learner_id=learner.module_id, signature=signature_gen(learner.module_id))
            state.outbox.send_learner_task(learner_id=learner.module_id, task=None, signature=signature_gen(learner.module_id))


class TestGoalGraph(GoalGraphBase, BaseAuxiliary):
    def step(self, state: GoalGraphState) -> GoalGraphState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Iterable[Goal], **kwargs) -> GoalGraphState:
        return self.process_and_send(self.on_goal_update.__name__, state, kwargs['signature'])

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs) -> GoalGraphState:
        return self.process_and_send(self.on_goal_request.__name__, state, kwargs['signature'])

    def message_all(self, state: GoalGraphState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for hl_reasoner in directory.internal.hl_reasoning:
            state.outbox.send_goals(receiver_id=hl_reasoner.module_id, goals=[], signature=signature_gen(hl_reasoner.module_id))
        for ll_reasoner in directory.internal.ll_reasoning:
            state.outbox.send_goals(receiver_id=ll_reasoner.module_id, goals=[], signature=signature_gen(ll_reasoner.module_id))


class TestMemory(MemoryBase, BaseAuxiliary):
    def step(self, state: MemoryState) -> MemoryState:
        state.step_counter += 1
        return self.process_and_send(self.step.__name__, state)

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        return self.process_and_send(self.on_memory_request.__name__, state, kwargs['signature'])

    def on_belief_update(self, state: MemoryState, sender: str, beliefs: Iterable[Belief], **kwargs) -> MemoryState:
        return self.process_and_send(self.on_belief_update.__name__, state, kwargs['signature'])

    def on_observation_update(self, state: MemoryState, sender: str, observations: Iterable[Observation], **kwargs) -> MemoryState:
        return self.process_and_send(self.on_observation_update.__name__, state, kwargs['signature'])

    def message_all(self, state: MemoryState, directory: Directory, func_name: str) -> None:
        signature_gen = self._signature_gen_factory(self.module_id, func_name)
        for learner in directory.internal.learning:
            state.outbox.send_memories(learner_id=learner.module_id, memories=[], signature=signature_gen(learner.module_id))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Provide state save directory as a command line argument.')

    save_dir = Path(argv[1])

    agent_id = 'test_agent_orchestrator'

    test_data = TestData(save_format='dill',
                         n_perceptors=3,
                         n_actuators=3,
                         n_ll_reasoners=3,
                         n_learners=3,
                         n_knowledge=3,
                         n_hl_reasoners=3,
                         n_goal_graphs=3,
                         n_memory=3
                         )

    orchestrator = Orchestrator(
        save_dir=save_dir,
        connector_cls=RabbitMQConnector,
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=test_data.step_frequency,
        exec_duration=test_data.exec_duration,
        save_format=test_data.save_format,
        resume=False,
        log_level=logging.DEBUG
    )

    orchestrator.add_agent(
        agent_id=agent_id,
        connector_cls=RabbitMQConnector,
        perceptors=[TestPerceptor(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.perceptors],
        actuators=[TestActuator(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.actuators],
        ll_reasoners=[TestLLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.ll_reasoners],
        learners=[TestLearner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.learners],
        knowledge=[TestKnowledge(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.knowledge],
        hl_reasoners=[TestHLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.hl_reasoners],
        goal_graphs=[TestGoalGraph(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.goal_graphs],
        memory=[TestMemory(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'sent_from': set(),
                'received_from': set()
            }
        ) for module_id in test_data.memory],
    )

    asyncio.run(orchestrator.arun(force_run=True, local_build=Path('C:\\phd\\mhagenta\\mhagenta').resolve()))
    check_results(
        agent_id='test_agent_orchestrator',
        save_dir=orchestrator[agent_id].save_dir,
        test_data=test_data,
        fields=['sent_from', 'received_from']
    )
