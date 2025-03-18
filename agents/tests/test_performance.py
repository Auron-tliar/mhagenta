import json
import sys
from abc import ABC, abstractmethod

import dill
from typing import Any, Iterable, Self, Iterator, Callable, Literal
from pathlib import Path
import logging
import asyncio
from sys import argv
import time

from mhagenta import State, Observation, ActionStatus, Goal, Belief, Orchestrator
from mhagenta.bases import *
from mhagenta.states import *
from mhagenta.utils.common import Directory
from mhagenta.core import RabbitMQConnector

from base import TestDataBase


class TestData(TestDataBase):
    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_perceptor(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_actuator(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_learner(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_knowledge(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_hl_reasoner(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_goal_graph(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_memory(self, module_id: str) -> dict[str, set]:
        return {}


class BaseAuxiliary(ABC):
    def on_callback(self, state: State) -> State:
        state.callback_counter += 1
        # self.message_all(state)
        return state

    @staticmethod
    @abstractmethod
    def message_all(state: State) -> None:
        pass


class TestLLReasoner(LLReasonerBase, BaseAuxiliary):
    def step(self, state: LLState) -> LLState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs) -> LLState:
        return self.on_callback(state)

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs) -> LLState:
        return self.on_callback(state)

    def on_goal_update(self, state: LLState, sender: str, goals: list[Goal], **kwargs) -> LLState:
        return self.on_callback(state)

    def on_model(self, state: LLState, sender: str, model: Any, **kwargs) -> LLState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    @staticmethod
    def message_all(state: LLState) -> None:
        for actuator in state.directory.internal.actuation:
            state.outbox.request_action(actuator_id=actuator.module_id)
        for perceptor in state.directory.internal.perception:
            state.outbox.request_observation(perceptor_id=perceptor.module_id)
        for learner in state.directory.internal.learning:
            state.outbox.request_model(learner_id=learner.module_id)
            state.outbox.send_learner_task(learner_id=learner.module_id, task=None)
        for goal_graph in state.directory.internal.goals:
            state.outbox.request_goals(goal_graph_id=goal_graph.module_id)
            state.outbox.send_goal_update(goal_graph_id=goal_graph.module_id, goals=[])
        for knowledge in state.directory.internal.knowledge:
            state.outbox.send_beliefs(knowledge_id=knowledge.module_id, observation=Observation(observation_type='none', content=None), beliefs=[])


class TestActuator(ActuatorBase, BaseAuxiliary):
    def step(self, state: ActuatorState) -> ActuatorState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: ActuatorState) -> None:
        for ll_reasoner in state.directory.internal.ll_reasoning:
            state.outbox.send_status(ll_reasoner_id=ll_reasoner.module_id, status=ActionStatus(status=None))


class TestPerceptor(PerceptorBase, BaseAuxiliary):
    def step(self, state: PerceptorState) -> PerceptorState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_request(self, state: PerceptorState, sender: str, **kwargs) -> PerceptorState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: PerceptorState) -> None:
        for ll_reasoner in state.directory.internal.ll_reasoning:
            state.outbox.send_observation(ll_reasoner_id=ll_reasoner.module_id, observation=Observation(observation_type='none', content=None))


class TestLearner(LearnerBase, BaseAuxiliary):
    def step(self, state: LearnerState) -> LearnerState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def on_task(self, state: LearnerState, sender: str, task: Any, **kwargs) -> LearnerState:
        return self.on_callback(state)

    def on_memories(self, state: LearnerState, sender: str, memories: Iterable[Belief | Observation], **kwargs) -> LearnerState:
        return self.on_callback(state)

    def on_model_request(self, state: LearnerState, sender: str, **kwargs) -> LearnerState:
        return self.on_callback(state)

    def message_all(self, state: LearnerState) -> None:
        for memory in state.directory.internal.memory:
            state.outbox.request_memories(memory_id=memory.module_id)
        for ll_reasoner in state.directory.internal.ll_reasoning:
            state.outbox.send_model(reasoner_id=ll_reasoner.module_id, model=None)
        for hl_reasoner in state.directory.internal.hl_reasoning:
            state.outbox.send_model(reasoner_id=hl_reasoner.module_id, model=None)


class TestKnowledge(KnowledgeBase, BaseAuxiliary):
    def step(self, state: KnowledgeState) -> KnowledgeState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_belief_request(self, state: KnowledgeState, sender: str, **kwargs) -> KnowledgeState:
        return self.on_callback(state)

    def on_belief_update(self, state: KnowledgeState, sender: str, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        return self.on_callback(state)

    def on_observed_beliefs(self, state: KnowledgeState, sender: str, observation: Observation, beliefs: Iterable[Belief], **kwargs) -> KnowledgeState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: KnowledgeState) -> None:
        for hl_reasoner in state.directory.internal.hl_reasoning:
            state.outbox.send_beliefs(hl_reasoner_id=hl_reasoner.module_id, beliefs=[])
        for memory in state.directory.internal.memory:
            state.outbox.send_belief_memories(memory_id=memory.module_id, beliefs=[])
            state.outbox.send_observations(memory_id=memory.module_id, observations=[])


class TestHLReasoner(HLReasonerBase, BaseAuxiliary):
    def step(self, state: HLState) -> HLState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_belief_update(self, state: HLState, sender: str, beliefs: Iterable[Belief], **kwargs) -> HLState:
        return self.on_callback(state)

    def on_goal_update(self, state: HLState, sender: str, goals: Iterable[Goal], **kwargs) -> HLState:
        return self.on_callback(state)

    def on_model(self, state: HLState, sender: str, model: Any, **kwargs) -> HLState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: HLState) -> None:
        for actuator in state.directory.internal.actuation:
            state.outbox.request_action(actuator_id=actuator.module_id)
        for knowledge in state.directory.internal.knowledge:
            state.outbox.request_beliefs(knowledge_id=knowledge.module_id)
            state.outbox.send_beliefs(knowledge_id=knowledge.module_id, beliefs=[])
        for goal_graph in state.directory.internal.goals:
            state.outbox.send_goals(goal_graph_id=goal_graph.module_id, goals=[])
        for learner in state.directory.internal.learning:
            state.outbox.request_model(learner_id=learner.module_id)
            state.outbox.send_learner_task(learner_id=learner.module_id, task=None)


class TestGoalGraph(GoalGraphBase, BaseAuxiliary):
    def step(self, state: GoalGraphState) -> GoalGraphState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_goal_update(self, state: GoalGraphState, sender: str, goals: Iterable[Goal], **kwargs) -> GoalGraphState:
        return self.on_callback(state)

    def on_goal_request(self, state: GoalGraphState, sender: str, **kwargs) -> GoalGraphState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: GoalGraphState) -> None:
        for hl_reasoner in state.directory.internal.hl_reasoning:
            state.outbox.send_goals(receiver_id=hl_reasoner.module_id, goals=[])
        for ll_reasoner in state.directory.internal.ll_reasoning:
            state.outbox.send_goals(receiver_id=ll_reasoner.module_id, goals=[])


class TestMemory(MemoryBase, BaseAuxiliary):
    def step(self, state: MemoryState) -> MemoryState:
        state.step_counter += 1
        self.message_all(state)
        return state

    def on_memory_request(self, state: MemoryState, sender: str, **kwargs) -> MemoryState:
        return self.on_callback(state)

    def on_belief_update(self, state: MemoryState, sender: str, beliefs: Iterable[Belief], **kwargs) -> MemoryState:
        return self.on_callback(state)

    def on_observation_update(self, state: MemoryState, sender: str, observations: Iterable[Observation], **kwargs) -> MemoryState:
        return self.on_callback(state)

    def on_first(self, state: State) -> LLState:
        state.start_time = time.perf_counter()
        return state

    def on_last(self, state: LLState) -> LLState:
        state.total_time = time.perf_counter() - state.start_time
        return state

    def message_all(self, state: MemoryState) -> None:
        for learner in state.directory.internal.learning:
            state.outbox.send_memories(learner_id=learner.module_id, memories=[])


def show_module_results(
        agent_id: str,
        module_id: str,
        save_dir: Path,
        save_format: Literal['json', 'dill'] = 'json'
) -> None:
    with open(Path(save_dir) / f'{agent_id}.{module_id}.{"sav" if save_format == "dill" else "json"}', 'rb') as f:
        state: dict = (dill if save_format == 'dill' else json).load(f)

    print(f'\t\t{module_id}...')
    print(f'\t\t\tStep count:     {state['step_counter']}\n'
          f'\t\t\tCallback count: {state['callback_counter']}\n'
          f'\t\t\tTotal count:    {state['step_counter'] + state['callback_counter']}\n'
          f'\t\t\tTotal time:     {state['total_time']}\n'
          )


def show_results(
        agent_id: str,
        save_dir: str | Path,
        test_data: TestDataBase
) -> None:
    print(f'================== Printing final statistics... ==================')

    print('\tPerceptors:')
    for perceptor in test_data.perceptors:
        show_module_results(
            agent_id=agent_id,
            module_id=perceptor,
            save_dir=save_dir,
            save_format='dill')

    print('\tActuators:')
    for actuator in test_data.actuators:
        show_module_results(
            agent_id=agent_id,
            module_id=actuator,
            save_dir=save_dir,
            save_format='dill')

    print('\tLow-level reasoners:')
    for ll_reasoner in test_data.ll_reasoners:
        show_module_results(
            agent_id=agent_id,
            module_id=ll_reasoner,
            save_dir=save_dir,
            save_format='dill')

    print('\tLearners:')
    for learner in test_data.learners:
        show_module_results(
            agent_id=agent_id,
            module_id=learner,
            save_dir=save_dir,
            save_format='dill')

    print('\tKnowledge:')
    for knowledge in test_data.knowledge:
        show_module_results(
            agent_id=agent_id,
            module_id=knowledge,
            save_dir=save_dir,
            save_format='dill')

    print('\tHigh-level reasoners:')
    for hl_reasoner in test_data.hl_reasoners:
        show_module_results(
            agent_id=agent_id,
            module_id=hl_reasoner,
            save_dir=save_dir,
            save_format='dill')

    print('\tGoal graphs:')
    for goal_graph in test_data.goal_graphs:
        show_module_results(
            agent_id=agent_id,
            module_id=goal_graph,
            save_dir=save_dir,
            save_format='dill')

    print('\tMemory:')
    for memory in test_data.memory:
        show_module_results(
            agent_id=agent_id,
            module_id=memory,
            save_dir=save_dir,
            save_format='dill')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Provide state save directory as a command line argument.')

    save_dir = Path(argv[1])
    agent_id = 'test_agent_performance'
    test_data = TestData(save_format='dill',
                         n_perceptors=1,
                         n_actuators=1,
                         n_ll_reasoners=1,
                         n_learners=1,
                         n_knowledge=1,
                         n_hl_reasoners=1,
                         n_goal_graphs=1,
                         n_memory=1,
                         step_frequency=0
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
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.perceptors],
        actuators=[TestActuator(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.actuators],
        ll_reasoners=[TestLLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.ll_reasoners],
        learners=[TestLearner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.learners],
        knowledge=[TestKnowledge(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.knowledge],
        hl_reasoners=[TestHLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.hl_reasoners],
        goal_graphs=[TestGoalGraph(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.goal_graphs],
        memory=[TestMemory(
            module_id=module_id,
            initial_state={
                'step_counter': 0,
                'callback_counter': 0,
                'start_time': 0.,
                'total_time': -1.
            }
        ) for module_id in test_data.memory],
        log_level=logging.ERROR
    )

    asyncio.run(orchestrator.arun(force_run=True, local_build=Path('C:\\phd\\mhagenta\\mhagenta').resolve()))
    show_results(
        agent_id='test_agent_performance',
        save_dir=orchestrator[agent_id].save_dir,
        test_data=test_data
    )
