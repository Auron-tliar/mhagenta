from abc import ABC, abstractmethod
from typing import Self, Iterator, Literal, Iterable
from pathlib import Path
import json
import dill


class TestDataBase(ABC):
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
                 n_rmq_perceptors: int | None = None,
                 n_rmq_actuators: int | None = None,
                 step_frequency: float = 0.5,
                 exec_duration: float = 10,
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

        self.step_frequency: float = step_frequency
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
    def hl_reasoners(self) -> Iterator:
        return self._hl_reasoners

    @property
    def goal_graphs(self) -> Iterator:
        return self._goal_graphs

    @property
    def memory(self) -> Iterator:
        return self._memory

    @property
    def expected_steps(self) -> int:
        return int(self.exec_duration / self.step_frequency)

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
        for hl_reasoner in self.hl_reasoners:
            expected[hl_reasoner] = self.expected_hl_reasoner(hl_reasoner)
        for goal_graph in self.goal_graphs:
            expected[goal_graph] = self.expected_goal_graph(goal_graph)
        for memory in self.memory:
            expected[memory] = self.expected_memory(memory)

        return expected

    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        pass

    def expected_perceptor(self, module_id: str) -> dict[str, set]:
        pass

    def expected_actuator(self, module_id: str) -> dict[str, set]:
        pass

    def expected_learner(self, module_id: str) -> dict[str, set]:
        pass

    def expected_knowledge(self, module_id: str) -> dict[str, set]:
        pass

    def expected_hl_reasoner(self, module_id: str) -> dict[str, set]:
        pass

    def expected_goal_graph(self, module_id: str) -> dict[str, set]:
        pass

    def expected_memory(self, module_id: str) -> dict[str, set]:
        pass

    @staticmethod
    def check_iterable(
            actual: Iterable,
            expected: Iterable,
            extra_ok: bool = False,
    ) -> tuple[bool, set, set]:
        actual_set = set(actual)
        expected_set = set(expected)
        result = (expected_set <= actual_set) if extra_ok else (actual_set == expected_set)
        if result:
            return result, set(), set()

        unexpected = actual_set.difference(expected_set)
        missed = expected_set.difference(actual_set)
        return result, unexpected, missed

    # @staticmethod
    # def check_scalar(
    #         actual: float | int | bool | str,
    #         expected: float | int | bool | str
    # ) -> bool:
    #     return actual == expected

def check_module_result(
        module_id: str,
        actual_state: dict[str, Iterable],
        expected_state: dict[str, Iterable],
        fields: Iterable[str],
        actual_steps: int,
        expected_steps: int,
        extra_ok: bool = False
) -> bool:
    print(f'\t\t{module_id}...')
    results = {key: TestDataBase.check_iterable(actual_state[key], expected_state[key], extra_ok) for key in fields}
    success = all([res[0] for res in results.values()]) and actual_steps >= expected_steps
    if success:
        print('\t\t\t...SUCCEEDED!')
        return True

    print('\t\t\t...FAILED!')
    if actual_steps < expected_steps:
        print(f'\t\t\t\tExpected at least {expected_steps} steps, got {actual_steps}')

    for key, value in results.items():
        if not value[0]:
            if value[1]:
                print(f'\t\t\t\tUnexpected \'{key}\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t\t".join([str(item) for item in value[1]])}')
            if value[2]:
                print(f'\t\t\t\tMissing \'{key}\' records:')
                print(f'\t\t\t\t{"\n\t\t\t\t\t".join([str(item) for item in value[2]])}')
    print()
    return False


def check_module(
        agent_id: str,
        module_id: str,
        save_dir: Path,
        expected_state: dict[str, set],
        expected_steps: int,
        fields: Iterable[str],
        extra_ok: bool = False,
        save_format: Literal['json', 'dill'] = 'json'
) -> int:
    with open(Path(save_dir) / f'{agent_id}.{module_id}.{"sav" if save_format == "dill" else "json"}', 'rb') as f:
        actual_state: dict = (dill if save_format == 'dill' else json).load(f)
    actual_steps = actual_state.pop('step_counter')
    return check_module_result(
        module_id=module_id,
        actual_state=actual_state,
        expected_state=expected_state,
        actual_steps=actual_steps,
        expected_steps=expected_steps,
        fields=fields,
        extra_ok=extra_ok
    )


def check_results(
        agent_id: str,
        save_dir: str | Path,
        test_data: TestDataBase,
        fields: Iterable[str],
        extra_ok: bool = False
) -> bool:
    print(f'================== Checking final states for {agent_id}... ==================')
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
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tActuators:')
    for actuator in test_data.actuators:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=actuator,
            save_dir=save_dir,
            expected_state=test_data.expected_actuator(actuator),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tLow-level reasoners:')
    for ll_reasoner in test_data.ll_reasoners:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=ll_reasoner,
            save_dir=save_dir,
            expected_state=test_data.expected_ll_reasoner(ll_reasoner),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tLearners:')
    for learner in test_data.learners:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=learner,
            save_dir=save_dir,
            expected_state=test_data.expected_learner(learner),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tKnowledge:')
    for knowledge in test_data.knowledge:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=knowledge,
            save_dir=save_dir,
            expected_state=test_data.expected_knowledge(knowledge),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tHigh-level reasoners:')
    for hl_reasoner in test_data.hl_reasoners:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=hl_reasoner,
            save_dir=save_dir,
            expected_state=test_data.expected_hl_reasoner(hl_reasoner),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tGoal graphs:')
    for goal_graph in test_data.goal_graphs:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=goal_graph,
            save_dir=save_dir,
            expected_state=test_data.expected_goal_graph(goal_graph),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print('\tMemory:')
    for memory in test_data.memory:
        total_tests += 1
        successful += check_module(
            agent_id=agent_id,
            module_id=memory,
            save_dir=save_dir,
            expected_state=test_data.expected_memory(memory),
            expected_steps=test_data.expected_steps,
            fields=fields,
            extra_ok=extra_ok,
            save_format='dill')

    print(f'TEST RESULTS: {successful}/{total_tests}')

    return successful == total_tests
