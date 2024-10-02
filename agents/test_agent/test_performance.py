import json
import sys

import dill
from typing import Any, Iterable, Self, Iterator, Literal
from pathlib import Path
import logging
import asyncio
from sys import argv

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
                 exec_duration: float = 10,
                 save_format: Literal['json', 'dill'] = 'json'
                 ) -> None:
        self.n_ll_reasoners = n_ll_reasoners if n_ll_reasoners is not None else self._n_ll_reasoners
        self.n_perceptors = n_perceptors if n_perceptors is not None else self._n_perceptors
        self.n_actuators = n_actuators if n_actuators is not None else self._n_actuators
        self.n_learners = n_learners if n_learners is not None else self._n_learners
        self.n_knowledge = n_knowledge if n_knowledge is not None else self._n_knowledge
        self.n_hl_reasoners = n_hl_reasoners if n_hl_reasoners is not None else self._n_hl_reasoners
        self.n_goal_graphs = n_goal_graphs if n_goal_graphs is not None else self._n_goal_graphs
        self.n_memory = n_memory if n_memory is not None else self._n_memory

        self._ll_reasoners = self.ModuleIterator('test_ll_reasoner_{}', self._n_ll_reasoners if n_ll_reasoners is None else n_ll_reasoners)
        self._perceptors = self.ModuleIterator('test_perceptor_{}', self._n_perceptors if n_perceptors is None else n_perceptors)
        self._actuators = self.ModuleIterator('test_actuator_{}', self._n_actuators if n_actuators is None else n_actuators)
        self._learners = self.ModuleIterator('test_learner_{}', self._n_learners if n_learners is None else n_learners)
        self._knowledge = self.ModuleIterator('test_knowledge_{}', self._n_knowledge if n_knowledge is None else n_knowledge)
        self._hl_reasoners = self.ModuleIterator('test_hl_reasoner_{}', self._n_hl_reasoners if n_hl_reasoners is None else n_hl_reasoners)
        self._goal_graphs = self.ModuleIterator('test_goal_graph_{}', self._n_goal_graphs if n_goal_graphs is None else n_goal_graphs)
        self._memory = self.ModuleIterator('test_memory_{}', self._n_memory if n_memory is None else n_memory)

        self.exec_duration: float = exec_duration
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


class TestLLReasoner(LLReasonerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_observation(self, state: State, sender: str, observation: Observation, **kwargs) -> Update:
        return state

    def on_action_status(self, state: State, sender: str, action_status: ActionStatus, **kwargs) -> Update:
        return state

    def on_goal_update(self, state: State, sender: str, goals: list[Goal], **kwargs) -> Update:
        return state

    def on_learning_status(self, state: State, sender: str, learning_status: Any, **kwargs) -> Update:
        return state

    def on_model(self, state: State, sender: str, model: Any, **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> LLOutbox:
        outbox = LLOutbox()
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator)
        for perceptor in directory.perception:
            outbox.request_observation(perceptor_id=perceptor)
        for learner in directory.learning:
            outbox.request_model(learner_id=learner)
            outbox.send_learner_task(learner_id=learner, task=None)
        for goal_graph in directory.goals:
            outbox.request_goals(goal_graph_id=goal_graph)
            outbox.send_goal_update(goal_graph_id=goal_graph, goals=[])
        for knowledge in directory.knowledge:
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[])
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, observations=[])
        return outbox


class TestActuator(ActuatorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> ActuatorOutbox:
        outbox = ActuatorOutbox()
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, status=ActionStatus(status=None))
        return outbox


class TestPerceptor(PerceptorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> PerceptorOutbox:
        outbox = PerceptorOutbox()
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_observation(ll_reasoner_id=ll_reasoner, observation=Observation(observation_type='none', value=None))
        return outbox


class TestLearner(LearnerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_task(self, state: State, sender: str, task: Any, **kwargs) -> Update:
        return state

    def on_memories(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        return state

    def on_model_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> LearnerOutbox:
        outbox = LearnerOutbox()
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, learning_status=None)
            outbox.send_model(ll_reasoner_id=ll_reasoner, model=None)
        return outbox


class TestKnowledge(KnowledgeBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> KnowledgeOutbox:
        outbox = KnowledgeOutbox()
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(knowledge_id=hl_reasoner, beliefs=[])
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, beliefs=[])
        return outbox


class TestHLReasoner(HLReasonerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return state

    def on_goal_update(self, state: State, sender: str, goals: Iterable[Goal], **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> HLOutbox:
        outbox = HLOutbox()
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory)
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator)
        for knowledge in directory.knowledge:
            outbox.request_beliefs(knowledge_id=knowledge)
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[])
        for goal_graph in directory.goals:
            outbox.send_goals(goal_graph_id=goal_graph, goals=[])
        return outbox


class TestGoalGraph(GoalGraphBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_goal_update(self, state: State, sender: str, goals: Iterable[Goal], **kwargs) -> Update:
        return state

    def on_goal_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> GoalGraphOutbox:
        outbox = GoalGraphOutbox()
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_goals(receiver=hl_reasoner, goals=[])
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_goals(receiver=ll_reasoner, goals=[])
        return outbox


class TestMemory(MemoryBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1
        return state, self.message_all(state.directory)

    def on_belief_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    def on_belief_update(self, state: State, sender: str, beliefs: Iterable[Belief], **kwargs) -> Update:
        return state

    def on_observation_request(self, state: State, sender: str, **kwargs) -> Update:
        return state

    def on_observation_update(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        return state

    @staticmethod
    def message_all(directory: Directory) -> MemoryOutbox:
        outbox = MemoryOutbox()
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(hl_reasoner_id=hl_reasoner, beliefs=[])
        for learner in directory.learning:
            outbox.send_observations(learner_id=learner, observations=[])
        return outbox


def get_module_results(agent_id: str, module_id: str, save_dir: Path, save_format: Literal['json', 'dill'] = 'json') -> int:
    with open(Path(save_dir) / f'{agent_id}.{module_id}.{"sav" if save_format == "dill" else "json"}', 'rb') as f:
        state: dict = (dill if save_format == 'dill' else json).load(f)
    steps = state['step_counter']
    return steps


def check_results(agent_id: str, save_dir: str | Path, test_data: TestData) -> None:
    print(f'================== RESULTS ==================')

    total_average = 0.
    total_num = 0

    print('\tPerceptors:')
    average = 0
    for perceptor in test_data.perceptors:
        steps = get_module_results(agent_id, perceptor, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{perceptor}: {steps}')
    total_average += average
    total_num += test_data.n_perceptors
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_perceptors}')

    print('\tActuators:')
    average = 0
    for actuator in test_data.actuators:
        steps = get_module_results(agent_id, actuator, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{actuator}: {steps}')
    total_average += average
    total_num += test_data.n_actuators
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_actuators}')

    print('\tLow-level reasoners:')
    average = 0
    for ll_reasoner in test_data.ll_reasoners:
        steps = get_module_results(agent_id, ll_reasoner, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{ll_reasoner}: {steps}')
    total_average += average
    total_num += test_data.n_ll_reasoners
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_ll_reasoners}')

    print('\tLearners:')
    average = 0
    for learner in test_data.learners:
        steps = get_module_results(agent_id, learner, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{learner}: {steps}')
    total_average += average
    total_num += test_data.n_learners
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_learners}')

    print('\tKnowledge:')
    average = 0
    for knowledge in test_data.knowledge:
        steps = get_module_results(agent_id, knowledge, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{knowledge}: {steps}')
    total_average += average
    total_num += test_data.n_knowledge
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_knowledge}')

    print('\tHigh-level reasoners:')
    average = 0
    for hl_reasoner in test_data.hl_reasoner:
        steps = get_module_results(agent_id, hl_reasoner, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{hl_reasoner}: {steps}')
    total_average += average
    total_num += test_data.n_hl_reasoners
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_hl_reasoners}')

    print('\tGoal graphs:')
    average = 0
    for goal_graph in test_data.goal_graphs:
        steps = get_module_results(agent_id, goal_graph, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{goal_graph}: {steps}')
    total_average += average
    total_num += test_data.n_goal_graphs
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_goal_graphs}')

    print('\tMemory:')
    average = 0
    for memory in test_data.memory:
        steps = get_module_results(agent_id, memory, save_dir, test_data.save_format)
        average += steps
        print(f'\t\t{memory}: {steps}')
    total_average += average
    total_num += test_data.n_memory
    print(f'\t\t-----------------\n\t\tAverage number of steps: {average / test_data.n_memory}')

    print(f'\t-----------------\n\tTotal average: {total_average / total_num} ({total_average / total_num / test_data.exec_duration} steps per second)')


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
    if len(sys.argv) < 2:
        raise AttributeError('Provide state save directory as a command line argument.')

    save_dir = Path(argv[1])

    test_data = TestData(save_format='dill')

    test_agent = MHARoot(
        agent_id='test_agent',
        connector_cls=RabbitMQConnector,
        perceptors=[TestPerceptor(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.perceptors],
        actuators=[TestActuator(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.actuators],
        ll_reasoners=[TestLLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.ll_reasoners],
        learners=[TestLearner(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.learners],
        knowledge=[TestKnowledge(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.knowledge],
        hl_reasoner=[TestHLReasoner(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.hl_reasoner],
        goal_graphs=[TestGoalGraph(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.goal_graphs],
        memory=[TestMemory(
            module_id=module_id,
            initial_state={
                'step_counter': 0
            }
        ) for module_id in test_data.memory],
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=0.,
        exec_duration=test_data.exec_duration,
        save_dir=save_dir,
        save_format=test_data.save_format,
        resume=False,
        log_level=logging.INFO
    )

    asyncio.run(run_agent(agent=test_agent, test_data=test_data, only_test=False))
